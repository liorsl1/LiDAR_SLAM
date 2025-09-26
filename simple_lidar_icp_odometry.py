import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

from conversions import read_points

import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation

class LidarOdometry:
    def __init__(self):
        self.node = rclpy.create_node('lidar_odometry_node')

        self.publisher_ = self.node.create_publisher(Odometry, 'pointcloud/odom', 10)

        self.subscription = self.node.create_subscription(
            PointCloud2,
            '/kitti/point_cloud',
            self.listener_callback,
            10)

        self.pc_odom_msg = Odometry()

        self.prev_cloud = None
        
        # Initialize an identity matrix as the initial odometry
        self.odometry = np.eye(4)

        self.transformation = np.eye(4)
        
        # Loop detection variables
        self.last_timestamp = None
        self.loop_detection_threshold = 5.0  # seconds - adjust as needed
        self.loop_just_detected = False

    def detect_new_loop(self, msg):
        """Detect if we've started a new loop by checking for time jumps."""
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        if self.last_timestamp is not None:
            time_diff = current_time - self.last_timestamp
            
            # Detect backward time jump (new loop) or large time gap
            if (time_diff < -self.loop_detection_threshold or time_diff > self.loop_detection_threshold) and not self.loop_just_detected:
                self.node.get_logger().info(f'Loop detected! Time jump: {time_diff:.2f}s - Resetting variables')
                self.loop_just_detected = True
                return True
            else:
                self.loop_just_detected = False
        
        self.last_timestamp = current_time
        return False

    def reset_variables(self):
        """Reset all variables for a new loop."""
        self.prev_cloud = None
        self.odometry = np.eye(4)
        self.transformation = np.eye(4)
        self.node.get_logger().info('Variables reset for new loop')

    def listener_callback(self, msg):
        # Check for new loop
        if self.detect_new_loop(msg):
            self.reset_variables()

        cloud = self.pointcloud2_to_pointcloud(msg)
        
        if self.prev_cloud is None:
            self.prev_cloud = cloud
            return

        self.transformation, inlier_rmse = self.perform_icp_point_to_plane(cloud, self.prev_cloud)

        # Multiply the current odometry with the new transformation matrix to update the odometry
        self.odometry = self.odometry @ self.transformation

        # Extract rotation matrix
        R = self.odometry[:3, :3]
        euler = self.rotation_to_euler(R)

        # Extract translation vector
        T = self.odometry[:3, 3]

        self.node.get_logger().info('LiDAR Odom: x: %f, y: %f, yaw: %f' % (T[0], T[1], euler[2]))

        self.publish_odometry(T[0], T[1], T[2], R, euler[2])

        self.prev_cloud = cloud

    def publish_odometry(self, x, y, z, R, yaw):
            quat_x, quat_y, quat_z, quat_w = self.rotation_to_quaternion(np.transpose(R))

            self.pc_odom_msg = Odometry()
            self.pc_odom_msg.header.stamp = self.node.get_clock().now().to_msg()
            self.pc_odom_msg.header.frame_id = "odom"
            self.pc_odom_msg.child_frame_id = ""
            self.pc_odom_msg.pose.pose.position.x = -x
            self.pc_odom_msg.pose.pose.position.y = y
            self.pc_odom_msg.pose.pose.position.z = z
            self.pc_odom_msg.pose.pose.orientation.x = quat_x
            self.pc_odom_msg.pose.pose.orientation.y = quat_y
            self.pc_odom_msg.pose.pose.orientation.z = quat_z
            self.pc_odom_msg.pose.pose.orientation.w = quat_w

            self.publisher_.publish(self.pc_odom_msg)
        
    def pointcloud2_to_pointcloud(self, msg):

        data = read_points(msg, skip_nans=True, field_names=("y", "x", "z"))
        
        """ Converts a ROS LaserScan message to an Open3D PointCloud object. """
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(data)
        downcloud = cloud.voxel_down_sample(voxel_size=1) # Smaller size results in more points
        return downcloud

    def perform_icp_point_to_plane(self, source, target):

        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))

        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))

        threshold = 1.0  # Distance threshold for RANSAC
        trans_init = self.transformation  # Initial transformation

        # Perform point-to-plane ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())

        return reg_p2p.transformation, reg_p2p.inlier_rmse

    def remove_outliers(self, point_cloud):
        cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
        return cloud

    
    def rotation_to_euler(self, R):
        """ Converts a rotation matrix to Euler angles. """

        r = Rotation.from_matrix(R)
        euler = r.as_euler('xyz', degrees=True)
        return euler   
    
    def rotation_to_quaternion(self, R):
        """ Converts a rotation matrix to Euler angles. """

        r = Rotation.from_matrix(R)
        # Convert to quaternion
        quaternion = r.as_quat()
        return quaternion
    
def main(args=None):
    rclpy.init(args=args)

    lidar_odometry_node = LidarOdometry()

    rclpy.spin(lidar_odometry_node.node)

    lidar_odometry_node.node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
