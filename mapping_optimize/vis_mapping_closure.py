import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2, LaserScan, Image, CameraInfo

import cv2
from cv_bridge import CvBridge

import tf2_ros

import numpy as np

from visual_hashmap import VisualHashMap
from graph_slam_data import GraphSLAMData
from visualizer import Visualizer
from features_utils import *

import time
import math
import random

from scipy.spatial.transform import Rotation

def add_noise_to_controls(v, w, std_dev_v, std_dev_w, time_step, max_std_multiplier=10):
    """
    Add Gaussian noise to the control inputs with logarithmic increase over time.

    Parameters:
    v (float): Linear velocity.
    w (float): Angular velocity.
    std_dev_v (float): Base standard deviation for linear velocity noise.
    std_dev_w (float): Base standard deviation for angular velocity noise.
    time_step (int): Current time step in the simulation.
    max_std_multiplier (float): Maximum multiplier for standard deviation.

    Returns:
    tuple: Noisy control inputs (v, w).
    """
    # Logarithmic increase of the standard deviation
    multiplier = min(np.log(1 + time_step), max_std_multiplier)
    current_std_dev_v = std_dev_v * multiplier
    current_std_dev_w = std_dev_w * multiplier

    noisy_v = np.random.normal(v, current_std_dev_v)
    noisy_w = np.random.normal(w, current_std_dev_w)
    return noisy_v, noisy_w


def update_pose_with_noisy_controls(pose, v, w, dt, std_dev_v, std_dev_w, time_step):
    """
    Update the robot pose based on noisy control inputs that increase over time.

    Parameters:
    pose (tuple): Current pose (x, y, theta).
    v (float): Linear velocity.
    w (float): Angular velocity.
    dt (float): Time step.
    std_dev_v (float): Initial standard deviation for linear velocity noise.
    std_dev_w (float): Initial standard deviation for angular velocity noise.
    time_step (int): Current time step in the simulation.

    Returns:
    tuple: Updated pose (x, y, theta).
    """
    noisy_v, noisy_w = add_noise_to_controls(v, w, std_dev_v, std_dev_w, time_step)

    # Motion model
    x, y, theta = pose
    if noisy_w != 0:
        x += -noisy_v/noisy_w * np.sin(theta) + noisy_v/noisy_w * np.sin(theta + noisy_w*dt)
        y += noisy_v/noisy_w * np.cos(theta) - noisy_v/noisy_w * np.cos(theta + noisy_w*dt)
    else:
        x += noisy_v * dt * np.cos(theta)
        y += noisy_v * dt * np.sin(theta)
    theta += noisy_w * dt
    theta = (theta + np.pi) % (2 * np.pi) - np.pi  # Normalize theta

    return x, y, theta, noisy_v, noisy_w


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector_node')

        self.visual_map = VisualHashMap(cell_size = 5)
        self.slam_data = GraphSLAMData()
        self.visualizer = Visualizer()

        self.bridge = CvBridge()

        self.odometry_subscription = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odometry_callback,
            10)

        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_scan_callback,
            10)

        self.image_subscription = self.create_subscription(
            Image,
            '/kitti/image/color/right',
            self.image_callback,
            10)


        self.laser_scan_publisher = self.create_publisher(LaserScan, 'scan', 10)

        map_update_timer_period = 0.5  # seconds
        self.map_update_timer = self.create_timer(map_update_timer_period, self.map_update_callback)
        
        self.std_dev_v = 0.1
        self.std_dev_w = 0.05

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.u = None
        self.scan = None
        self.odom = None
        self.noisy_odom = None
        self.image = None
        self.camera_h_fov = None
        self.camera_v_fov = None
        self.image_width = None
        self.lidar_fov = None
        self.lidar_angular_resolution = None
        self.overlapping_sector = None

        self.initial_pose_set = False
        self.initial_pose = None

        self.landmarks = []
        self.loop_closure_landmarks = []

        self.frame_count = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.angular_offset = 1.5708
        if self.angular_offset is not None:
            self.get_logger().info(f'Angular Offset: {math.degrees(self.angular_offset)} degrees')

        self.horizontal_offset, self.vertical_offset = -7.631618e-02, -2.717806e-01
        if self.vertical_offset is not None:
            self.get_logger().info(f'Vertical Offset: {self.vertical_offset} meters')
            self.get_logger().info(f'Horizontal Offset: {self.horizontal_offset} meters')

        # Create the CameraInfo message
        camera_info_msg = self.create_camera_info()

    def laser_scan_callback(self, msg):
        self.scan = msg
        self.scan.range_max = 500.0
        self.lidar_fov = msg.angle_max - msg.angle_min
        self.lidar_angular_resolution = msg.angle_increment

    def odometry_callback(self, msg):
        self.odom = msg

        # Set the initial pose
        if not self.initial_pose_set:
            self.initial_pose = self.odom_to_pose()
            self.initial_pose_set = True

        self.u = [msg.twist.twist.linear.x + random.gauss(0.0, 0.5), msg.twist.twist.angular.z + random.gauss(0.0, 0.5)]

    def image_callback(self, msg):
        if self.image_width is None or self.overlapping_sector is None or self.u is None:
            return

        self.image = msg

        if self.frame_count % 1 == 0:

            # Mapping p(m | z, x) is done here:

            # Feature-based measurement model P(z | m, x) -------------------------
            # Extract features from the image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            keypoints, descriptors = extract_features_dbscan(cv_image)

            vertical_band_center = self.image_height / 2  # Centered vertically, adjust as needed
            vertical_band_height = self.image_height  # Height of the band in pixels, adjust as needed
            keypoints, descriptors = filter_keypoints_in_vertical_band(
                keypoints, descriptors, self.image_height, vertical_band_center, vertical_band_height)

            # Calculate angles from center of camera
            feature_angles = [convert_x_to_angle(kp.pt[0], self.image_width, self.camera_h_fov) for kp in keypoints]

            # Obtain angles and ranges from laser scan
            laser_ranges, laser_angles, _,  max_range, _ = lidar_scan(self.scan)
        
            # Sensor Fusion! Fuse visual features with lidar data
            distances, angles, descriptors, indices = match_features_to_laserscan(feature_angles, laser_angles, laser_ranges, descriptors, self.overlapping_sector, max_lidar_range = max_range)
            self.view_features(cv_image, keypoints, indices)
            # ----------------------------------------------------------

            # Convert from polar to cartesian coordinates
            positions = [compute_xy_location(distance, angle, self.horizontal_offset) for distance, angle in zip(distances, angles)]
            
            # Convert from local to global frame 
            robot_pose = self.get_pose()
            # Add noise to pose
            # self.robot_x, self.robot_y, self.robot_yaw, noisy_v, noisy_w = update_pose_with_noisy_controls(robot_pose, self.u[0], self.u[1], 0.1, self.std_dev_v, self.std_dev_w, self.frame_count / 5)
            self.robot_x, self.robot_y, self.robot_yaw = robot_pose
            positions = [transform_point_to_global_frame(pos[0], pos[1], self.robot_x, self.robot_y, self.robot_yaw) for pos in positions]

            current_time = self.get_clock().now()
            # control_index = self.slam_data.add_control_and_pose(noisy_v, noisy_w, self.robot_x, self.robot_y, self.robot_yaw, current_time.nanoseconds)
            control_index = self.slam_data.add_control_and_pose(self.u[0], self.u[1], self.robot_x, self.robot_y, self.robot_yaw, current_time.nanoseconds)
            
            # Build the temporary visual map
            visual_map = [(pos[0], pos[1], desc, r, phi) for pos, desc, r, phi in zip(positions, descriptors, distances, angles)]
            seen = []
            
            # Do correspondence matching (loop closure detection) c_t_hat = argmax_c_t (p(z_t| c, m, z)) ---------------------
            self.loop_closure_landmarks = []
            for feature in (visual_map):
                landmark = self.visual_map.match_and_update(robot_pose, feature, self.overlapping_sector, max_range, localize = False)

                _, _, _, r, phi = feature
                pos_x, pos_y, desc, landmark_index = landmark

                if landmark_index in seen:
                    continue
                else:
                    seen.append(landmark_index)

                # if it is a new landmark, add it
                if landmark_index == len(self.landmarks):
                    self.landmarks = self.visual_map.get_all_features()
                    print("New LM", len(self.landmarks))
                    # self.landmarks.append((pos_x, pos_y, desc))
                    self.slam_data.add_landmark(pos_x, pos_y, desc)
                else:
                    self.loop_closure_landmarks.append((pos_x, pos_y))

                self.visualizer.update_loop_closure(self.loop_closure_landmarks)
                self.slam_data.add_observation(control_index, landmark_index, r, phi)
            # ------------------------------------------------------------------------      
                    
        self.frame_count += 1

    def create_camera_info(self):
        msg = CameraInfo()

        # Set the camera matrix (K)
        msg.k = [9.037596e+02, 0.000000e+00, 6.957519e+02,  # Row 1
                         0.000000e+00, 9.019653e+02, 2.242509e+02,  # Row 2
                         0.000000e+00, 0.000000e+00, 1.000000e+00]  # Row 3

        # Set the distortion parameters (D)
        msg.d = [-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]

        # Set the rectified camera matrix (P)
        msg.p = [7.215377e+02, 0.000000e+00, 6.095593e+02, -3.395242e+02,  # Row 1
                         0.000000e+00, 7.215377e+02, 1.728540e+02, 2.199936e+00,   # Row 2
                         0.000000e+00, 0.000000e+00, 1.000000e+00, 2.729905e-03]   # Row 3

        # Set other necessary fields (adjust these as needed)
        msg.width = 1242  # Width of the image
        msg.height = 375  # Height of the image
        msg.distortion_model = "plumb_bob"  # Common distortion model

        width = msg.width
        height = msg.height
        f_x = msg.k[0]  # Focal length in x direction
        f_y = msg.k[4]  # Focal length in y direction

        fov_horizontal = 2 * math.atan(width / (2 * f_x))
        fov_vertical = 2 * math.atan(height / (2 * f_y))

        fov_horizontal_degrees = math.degrees(fov_horizontal)
        fov_vertical_degrees = math.degrees(fov_vertical)

        self.camera_h_fov = fov_horizontal_degrees
        self.camera_v_fov = fov_vertical_degrees
        self.image_width = width
        self.image_height = height

        #self.get_logger().info(
        #    f'Horizontal FOV: {fov_horizontal_degrees:.2f} degrees, '
        #    f'Vertical FOV: {fov_vertical_degrees:.2f} degrees'
        #)

        if self.angular_offset is None:
            return

        self.overlapping_sector = self.calculate_overlapping_sector(self.camera_h_fov, self.angular_offset)
        self.get_logger().info(
            f'Adjusted range: {self.overlapping_sector}' 
        )

    def map_update_callback(self):
        if self.odom is None:
            return

        # self.robot_x, self.robot_y, self.robot_yaw  = self.get_pose()
        pred_robot_x, pred_robot_y, pred_robot_yaw = None, None, None
        self.visualizer.update(self.landmarks, self.robot_x, self.robot_y, self.robot_yaw, pred_robot_x, pred_robot_y, pred_robot_yaw)
        self.visualizer.update_loop_closure(self.loop_closure_landmarks)

    def view_features(self, cv_image, keypoints, indices):
        green = []
        red = []
        # Draw keypoints within FOV in green
        for idx in indices:
            green.append(keypoints[idx])
        cv_image_with_keypoints = cv2.drawKeypoints(cv_image, green, None, color=(0,255,0), flags=0)
        # Draw keypoints outside FOV in red
        for i, kp in enumerate(keypoints):
            if i not in indices:
                red.append(keypoints[i])
        cv_image_with_keypoints = cv2.drawKeypoints(cv_image_with_keypoints, red, None, color=(0,0,255), flags=0)

        scale_factor = 0.5  # Downscale by half
        new_width = int(cv_image_with_keypoints.shape[1] * scale_factor)
        new_height = int(cv_image_with_keypoints.shape[0] * scale_factor)
        new_dimensions = (new_width, new_height)

        resized_image = cv2.resize(cv_image_with_keypoints, new_dimensions, interpolation=cv2.INTER_AREA)

            
        cv2.imshow("Detected Features", resized_image)
        cv2.waitKey(1)
        
    def odom_to_pose(self):
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        yaw = self.quat_to_yaw(self.odom.pose.pose.orientation)
        return (x, y, yaw)

    def get_pose(self):
        # Check if the initial pose is set
        if self.initial_pose_set:
            # Extract the current pose from odometry
            current_pose = self.odom_to_pose()
            # Transform the current pose relative to the initial pose
            transformed_pose = self.transform_pose(current_pose)
            return transformed_pose
        else:
            return (0.0, 0.0, 0.0)  # Default pose if initial pose not set

    def transform_pose(self, pose):
        # Transform the pose relative to the initial pose
        x, y, yaw = pose
        init_x, init_y, init_yaw = self.initial_pose

        # Adjust position
        x -= init_x
        y -= init_y

        # Adjust orientation
        yaw = (yaw - init_yaw + math.pi) % (2 * math.pi) - math.pi

        return (x, y, yaw)

    def quat_to_yaw(self, quat):
        rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        return rot.as_euler('xyz')[2]

    def calculate_angular_offset(self, frame1, frame2, reference_frame):
        # Wait for the transform to be available
        while not self.tf_buffer.can_transform(reference_frame, frame1, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
            self.get_logger().info('Waiting for transform from {} to {}'.format(reference_frame, frame1))
            rclpy.spin_once(self, timeout_sec=0.1)

        while not self.tf_buffer.can_transform(reference_frame, frame2, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
            self.get_logger().info('Waiting for transform from {} to {}'.format(reference_frame, frame2))
            rclpy.spin_once(self, timeout_sec=0.1)

        try:
            # Get transforms to a common reference frame
            trans1 = self.tf_buffer.lookup_transform(reference_frame, frame1, rclpy.time.Time())
            trans2 = self.tf_buffer.lookup_transform(reference_frame, frame2, rclpy.time.Time())

            # Get yaw angles from quaternions
            yaw1 = self.quat_to_yaw(trans1.transform.rotation)
            yaw2 = self.quat_to_yaw(trans2.transform.rotation)

            # Calculate yaw (angular offset)
            yaw_offset = yaw2 - yaw1

            return yaw_offset  # Convert to degrees for readability
        except Exception as e:
            self.get_logger().error(f'Error calculating offset: {e}')
            return None

    def calculate_overlapping_sector(self, camera_fov, angular_offset):
        # Camera's angular range relative to its own forward direction
        camera_range = (-camera_fov / 2, camera_fov / 2)

        # Adjust by the angular offset to align with the range finder's frame
        overlapping_sector = (math.radians(camera_range[0] + angular_offset), math.radians(camera_range[1] + angular_offset))

        return overlapping_sector

    def get_vertical_offset(self, source_frame, target_frame):
        try:
            # Wait for transform to be available
            while not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time()):
                self.get_logger().info('Waiting for transform')
                rclpy.spin_once(self, timeout_sec=0.1)

            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            return transform.transform.translation.z
        except Exception as e:
            self.get_logger().error('Error getting vertical offset: {}'.format(e))
            return None

    def get_offset(self, source_frame, target_frame):
        try:
            # Wait for transform to be available
            while not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time()):
                self.get_logger().info('Waiting for transform')
                rclpy.spin_once(self, timeout_sec=0.1)

            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            
            # Extract the horizontal and vertical offsets
            offset_y = transform.transform.translation.y  
            offset_z = transform.transform.translation.z

            return offset_y, offset_z

        except Exception as e:
            self.get_logger().error('Error getting offset: {}'.format(e))
            return None, None

def main(args=None):
    try:
        rclpy.init(args=args)
        data_collector_node = DataCollectorNode()
        rclpy.spin(data_collector_node)
    except KeyboardInterrupt:
        pass  # Handle Ctrl-C interruption
    finally:
        # Prompt to save the map happens here
        save_yn = input("Do you want to save the data (y/n)? ")
        if save_yn.lower() in ["y", "yes"]:
            name = input("Enter dataset name (default: mydata.pkl): ")
            if name == "":
                name = "mydata.pkl"
            data_collector_node.slam_data.save_to_file(name)
            print("Data saved!")
        
        # Clean up ROS2 state
        data_collector_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
