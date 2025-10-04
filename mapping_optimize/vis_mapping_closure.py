import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2, LaserScan, Image, CameraInfo, PointField
from sensor_msgs_py import point_cloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer

import cv2
from cv_bridge import CvBridge

import tf2_ros

import numpy as np
import open3d as o3d
from visual_hashmap import VisualHashMap
from graph_slam_data import GraphSLAMData
from visualizer import Visualizer
from features_utils import *
from conversions import read_points

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
        x += -noisy_v / noisy_w * np.sin(theta) + noisy_v / noisy_w * np.sin(
            theta + noisy_w * dt
        )
        y += noisy_v / noisy_w * np.cos(theta) - noisy_v / noisy_w * np.cos(
            theta + noisy_w * dt
        )
    else:
        x += noisy_v * dt * np.cos(theta)
        y += noisy_v * dt * np.sin(theta)
    theta += noisy_w * dt
    theta = (theta + np.pi) % (2 * np.pi) - np.pi  # Normalize theta

    return x, y, theta, noisy_v, noisy_w


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__("data_collector_node")

        self.visual_map = VisualHashMap(cell_size=5)
        self.slam_data = GraphSLAMData()
        self.visualizer = Visualizer()

        self.bridge = CvBridge()

        self.odometry_subscription = self.create_subscription(
            Odometry, "/odometry/filtered", self.odometry_callback, 10
        )

        # Synchronized subscribers
        self.img_sub = Subscriber(self, Image, "/kitti/image/color/left")
        self.scan_sub = Subscriber(self, LaserScan, "/scan")
        self.pc_sub = Subscriber(self, PointCloud2, "/kitti/point_cloud")

        self.sync = ApproximateTimeSynchronizer(
            [self.img_sub, self.scan_sub, self.pc_sub],
            queue_size=30,
            slop=0.08,  # 80 ms tolerance
            allow_headerless=False,
        )
        self.sync.registerCallback(self.synced_callback)

        self.laser_scan_publisher = self.create_publisher(LaserScan, "scan", 10)
        self.seg_points_pub = self.create_publisher(
            PointCloud2, "segmented_lidar_points", 10
        )

        map_update_timer_period = 0.5  # seconds
        self.map_update_timer = self.create_timer(
            map_update_timer_period, self.map_update_callback
        )

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
        self.current_cloud = None
        self.initial_pose_set = False
        self.initial_pose = None

        self.landmarks = []
        self.loop_closure_landmarks = []

        self.frame_count = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.angular_offset = 1.5708
        if self.angular_offset is not None:
            self.get_logger().info(
                f"Angular Offset: {math.degrees(self.angular_offset)} degrees"
            )

        self.horizontal_offset, self.vertical_offset = -7.631618e-02, -2.717806e-01
        if self.vertical_offset is not None:
            self.get_logger().info(f"Vertical Offset: {self.vertical_offset} meters")
            self.get_logger().info(
                f"Horizontal Offset: {self.horizontal_offset} meters"
            )

        # Create the CameraInfo message
        camera_info_msg = self.create_camera_info()
        self.init_transforms()
        # Segmentation
        self.enable_segmentation = True
        self.seg_model = None
        self.last_seg_results = []  # list of (class_name, mask, conf)

    def load_segmentation_model(self):
        if self.seg_model is None:
            try:
                from ultralytics import YOLO

                # Use lightweight segmentation model; change to 'yolov8x-seg.pt' for higher accuracy
                self.seg_model = YOLO("yolov8s-seg.pt")
                self.get_logger().info("YOLOv8 segmentation model loaded")
            except Exception as e:
                self.get_logger().error(f"Failed to load YOLOv8 model: {e}")
                self.enable_segmentation = False

    def segment_image_yolov8(self, cv_image, conf_thres=0.25):
        """Run YOLOv8 segmentation on BGR image, return list of (class_name, mask, conf)."""
        if not self.enable_segmentation:
            return []
        self.load_segmentation_model()
        if self.seg_model is None:
            return []
        try:
            results = self.seg_model.predict(
                source=cv_image, imgsz=640, conf=conf_thres, verbose=False
            )
        except Exception as e:
            self.get_logger().error(f"Segmentation inference failed: {e}")
            return []
        out = []
        if not results:
            return out
        r = results[0]
        names = r.names
        if r.masks is None:
            return out
        # r.masks.data shape: (N, H, W) torch tensor (boolean/float)
        import torch

        masks = r.masks.data  # (N,H,W)
        boxes = r.boxes  # (N, ...)
        w, h = cv_image.shape[1], cv_image.shape[0]
        for i in range(masks.shape[0]):
            cls_id = int(boxes.cls[i].item()) if boxes is not None else -1
            conf = float(boxes.conf[i].item()) if boxes is not None else 0.0
            if conf < 0.5:
                continue
            class_name = (
                names.get(cls_id, str(cls_id))
                if isinstance(names, dict)
                else names[cls_id]
            )
            mask_np = masks[i].detach().cpu().numpy().astype(np.uint8)  # 0/1 mask
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            # swap y,x to x,y
            mask_np = np.transpose(mask_np, (1, 0))
            out.append((class_name, mask_np, conf))

        print(
            f"Detected {len(out)} objects with confidence >= {conf_thres}, classes: {[c for c,_,_ in out]}"
        )
        return out

    def visualize_segmentations(self, cv_image, seg_results, alpha=0.5):
        if not seg_results:
            return cv_image
        overlay = cv_image.copy()
        h, w = cv_image.shape[:2]
        rng = np.random.default_rng(12345)
        for class_name, mask, conf in seg_results:
            if mask.shape[0] != h or mask.shape[1] != w:
                # Resize mask to image size if needed
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            color = rng.integers(0, 255, size=3).tolist()
            overlay[mask > 0] = (
                0.6 * np.array(color) + 0.6 * overlay[mask > 0]
            ).astype(np.uint8)
            # Put class label at centroid
            ys, xs = np.where(mask > 0)
            if xs.size:
                cx, cy = int(xs.mean()), int(ys.mean())
                cv2.putText(
                    overlay,
                    f"{class_name}:{conf:.2f}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        blended = cv2.addWeighted(cv_image, 1 - alpha, overlay, alpha, 0)
        return blended

    def init_transforms(self):
        # left cam & velo
        calib = [
            "P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00",
            "P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00",
            "P2: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485728000000e+01 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-03",
            "P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-03",
            "R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01",
            "Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01",
            "Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01",
        ]
        # P2 (3 x 4) for left eye
        self.P2 = np.matrix(
            [float(x) for x in calib[2].strip("\n").split(" ")[1:]]
        ).reshape(3, 4)
        R0_rect = np.matrix(
            [float(x) for x in calib[4].strip("\n").split(" ")[1:]]
        ).reshape(3, 3)
        # Add a 1 in bottom-right, reshape to 4 x 4
        R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
        self.R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
        Tr_velo_to_cam = np.matrix(
            [float(x) for x in calib[5].strip("\n").split(" ")[1:]]
        ).reshape(3, 4)
        self.T_velo_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)

    def pointcloud2_to_pointcloud(self, msg):

        data = read_points(msg, skip_nans=True, field_names=("y", "x", "z"))

        """ Converts a ROS LaserScan message to an Open3D PointCloud object. """
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(data)
        downcloud = cloud.voxel_down_sample(
            voxel_size=1
        )  # Smaller size results in more points
        return downcloud

    def pointcloud2_to_array(self, pc_msg: PointCloud2):
        pts = []
        for p in point_cloud2.read_points(
            pc_msg, field_names=("x", "y", "z"), skip_nans=True
        ):
            pts.append([p[0], p[1], p[2]])
        if not pts:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32)

    def synced_callback(self, img_msg: Image, scan_msg: LaserScan, pc_msg: PointCloud2):
        # Update latest sensor data atomically
        self.scan = scan_msg
        self.current_cloud = self.pointcloud2_to_array(pc_msg)

        # Store synchronized timestamp for publishing
        self.current_msg_timestamp = img_msg.header.stamp

        # Reuse existing logic (image_callback) by passing the image
        self.image_callback(img_msg)

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

        self.u = [
            msg.twist.twist.linear.x + random.gauss(0.0, 0.5),
            msg.twist.twist.angular.z + random.gauss(0.0, 0.5),
        ]

    def project_cloud_to_image(self, image_shape):
        """
        cloud: (N,3) numpy array in velodyne frame
        P: (3,4) projection matrix
        image_shape: (H,W) for filtering

        returns: (u,v,valid_mask)
        """
        cloud = self.current_cloud
        P = self.P2 @ self.R0_rect @ self.T_velo_cam  # (3,4)
        N = cloud.shape[0]

        # Homogenize cloud
        pts_h = np.hstack((cloud, np.ones((N, 1))))  # (N,4)

        # Project
        pts_2d = (P @ pts_h.T).T  # (N,3)

        # Normalize
        u = pts_2d[:, 0] / pts_2d[:, 2]
        v = pts_2d[:, 1] / pts_2d[:, 2]

        # Validity: in front of camera + inside image bounds
        H, W = image_shape
        valid = (pts_2d[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        # only keep points that are within band of half height of image center
        vertical_band_center = H / 2
        vertical_band_height = H  # height of the band in pixels, adjust as needed
        valid &= (v > vertical_band_center - vertical_band_height / 2) & (
            v < vertical_band_center + vertical_band_height / 2
        )
        # return u,v,valid indexes
        return (
            np.array([u.astype(np.int32)[valid], v.astype(np.int32)[valid]]),
            np.where(valid)[0],
        )

    def match_keypoints_to_cloud(self, keypoints, uv_proj, valid_ids, pixel_radius=4.0):
        from scipy.spatial import cKDTree

        """
        Match image keypoints to nearest projected LiDAR points.
        Args:
            keypoints   : list of cv2.KeyPoint
            uv_proj     : (M,2) float array of projected LiDAR pixels
            valid_ids   : (M,) indices into self.current_cloud
            pixel_radius: max pixel distance to accept
        Returns:
            matches: list of (kp_index, cloud_point_index, cloud_point_xyz, pixel_dist)
        """
        # Early exits
        if uv_proj is None or len(keypoints) == 0:
            return []

        # Handle empty
        if uv_proj.size == 0:
            return []

        uv_proj = uv_proj[0]
        # Accept several shapes:
        # (2, M)  -> transpose to (M,2)
        # (M, 2)  -> ok
        # (2,)    -> single point -> (1,2)
        # Anything else -> reject
        if uv_proj.ndim == 1:
            if uv_proj.shape[0] == 2:
                uv_proj = uv_proj.reshape(1, 2)
            else:
                print(
                    f"[match_keypoints_to_cloud] Rejecting 1D uv_proj shape {uv_proj.shape}"
                )
                return []
        elif uv_proj.ndim == 2:
            if uv_proj.shape[0] == 2 and uv_proj.shape[1] != 2:
                # Provided as (2, M)
                uv_proj = uv_proj.T
            elif uv_proj.shape[1] != 2:
                print(
                    f"[match_keypoints_to_cloud] Invalid 2D uv_proj shape {uv_proj.shape}; need (*,2)"
                )
                return []
        else:
            print(f"[match_keypoints_to_cloud] Invalid ndim {uv_proj.ndim}")
            return []

        # Cast to float32 for KD-tree
        if uv_proj.dtype != np.float32 and uv_proj.dtype != np.float64:
            uv_proj = uv_proj.astype(np.float32)

        # Build KD-tree
        try:
            tree = cKDTree(uv_proj)
        except Exception as e:
            print(f"[match_keypoints_to_cloud] KD-tree build error: {e}")
            return []

        matches = []
        used_cloud = set()
        for i, kp in enumerate(keypoints):
            kp_uv = np.array(kp.pt, dtype=np.float32)
            dist, idx = tree.query(kp_uv, k=1)
            if dist <= pixel_radius:
                cloud_global_idx = valid_ids[idx]
                if cloud_global_idx in used_cloud:
                    # Enforce one-to-one (skip if already taken). Remove this
                    # check if many-to-one is acceptable.
                    continue
                used_cloud.add(cloud_global_idx)
                cloud_xyz = self.current_cloud[cloud_global_idx]
                matches.append((i, cloud_global_idx, cloud_xyz, float(dist)))
        return matches

    def visualize_keypoints_and_projected(
        self, cv_image, keypoints, uv_proj, window_name="KP_vs_LiDAR"
    ):
        """Overlay image keypoints and projected LiDAR pixels.

        keypoints: list of cv2.KeyPoint
        uv_proj: can be (N,2) or (2,N) or empty.
        Colors:
          - Keypoints: green circles.
          - Projected LiDAR points: magenta small dots.
        """
        if cv_image is None:
            return
        # Normalize uv_proj shape to (N,2)
        if uv_proj is None:
            uv_proj = np.empty((0, 2), dtype=np.float32)

        vis = cv_image.copy()
        # Draw keypoints
        for kp in keypoints:
            u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
            cv2.circle(vis, (u, v), 3, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        # Resize for display consistency
        scale = 0.6
        vis_small = cv2.resize(
            vis,
            (int(vis.shape[1] * scale), int(vis.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imshow(window_name, vis_small)
        cv2.waitKey(1)

    def image_callback(self, msg):
        if (
            self.image_width is None
            or self.overlapping_sector is None
            or self.u is None
        ):
            return

        self.image = msg

        if self.frame_count % 1 == 0:

            # Mapping p(m | z, x) is done here:

            # Feature-based measurement model P(z | m, x) -------------------------
            # Extract features from the image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Optional segmentation
            if self.enable_segmentation:
                self.last_seg_results = self.segment_image_yolov8(cv_image)
                # seg_vis = self.visualize_segmentations(cv_image, self.last_seg_results)
            keypoints, descriptors = extract_features_dbscan(cv_image)

            vertical_band_center = (
                self.image_height / 2
            )  # Centered vertically, adjust as needed
            vertical_band_height = (
                self.image_height
            )  # Height of the band in pixels, adjust as needed
            keypoints, descriptors = filter_keypoints_in_vertical_band(
                keypoints,
                descriptors,
                self.image_height,
                vertical_band_center,
                vertical_band_height,
            )

            # Calculate angles from center of camera
            feature_angles = [
                convert_x_to_angle(kp.pt[0], self.image_width, self.camera_h_fov)
                for kp in keypoints
            ]

            # Obtain angles and ranges from laser scan
            laser_ranges, laser_angles, _, max_range, _ = lidar_scan(self.scan)
            # get projected pixels of relevant lidar frame
            projected_pixels, valid_3d_ids = self.project_cloud_to_image(
                cv_image.shape[:2]
            )
            projected_pixels = np.squeeze(projected_pixels).T
            # Visualize before matching
            # self.visualize_keypoints_and_projected(cv_image, keypoints, projected_pixels)
            # Match between projected pixels to keypoints from image
            # matches = self.match_keypoints_to_cloud(keypoints, valid_3d_ids, pixel_radius=20.0)
            # match between lidar points and segmented pixels - publish immediately for synchronization
            associations, segment_pixels = self.associate_segments_with_points(
                projected_pixels, valid_3d_ids
            )
            self.publish_segmented_points(associations)

            # Fuse visual features with lidar data
            distances, angles, descriptors, indices = match_features_to_laserscan(
                feature_angles,
                laser_angles,
                laser_ranges,
                descriptors,
                self.overlapping_sector,
                max_lidar_range=max_range,
            )
            # self.view_features(cv_image, keypoints, projected_pixels, indices, segment_pixels)
            # ----------------------------------------------------------

            # Convert from polar to cartesian coordinates
            positions = [
                compute_xy_location(distance, angle, self.horizontal_offset)
                for distance, angle in zip(distances, angles)
            ]

            # Convert from local to global frame
            robot_pose = self.get_pose()
            # Add noise to pose
            # self.robot_x, self.robot_y, self.robot_yaw, noisy_v, noisy_w = update_pose_with_noisy_controls(robot_pose, self.u[0], self.u[1], 0.1, self.std_dev_v, self.std_dev_w, self.frame_count / 5)
            self.robot_x, self.robot_y, self.robot_yaw = robot_pose
            positions = [
                transform_point_to_global_frame(
                    pos[0], pos[1], self.robot_x, self.robot_y, self.robot_yaw
                )
                for pos in positions
            ]

            # Use synchronized timestamp for consistency
            sync_time_ns = (
                self.current_msg_timestamp.sec * 1000000000
                + self.current_msg_timestamp.nanosec
                if hasattr(self, "current_msg_timestamp")
                else self.get_clock().now().nanoseconds
            )
            # control_index = self.slam_data.add_control_and_pose(noisy_v, noisy_w, self.robot_x, self.robot_y, self.robot_yaw, sync_time_ns)
            control_index = self.slam_data.add_control_and_pose(
                self.u[0],
                self.u[1],
                self.robot_x,
                self.robot_y,
                self.robot_yaw,
                sync_time_ns,
            )

            # Build the temporary visual map
            visual_map = [
                (pos[0], pos[1], desc, r, phi)
                for pos, desc, r, phi in zip(positions, descriptors, distances, angles)
            ]
            seen = []

            # Do correspondence matching (loop closure detection) c_t_hat = argmax_c_t (p(z_t| c, m, z)) ---------------------
            self.loop_closure_landmarks = []
            for feature in visual_map:
                landmark = self.visual_map.match_and_update(
                    robot_pose,
                    feature,
                    self.overlapping_sector,
                    max_range,
                    localize=False,
                )

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

    def associate_segments_with_points(self, projected_pixels, valid_3d_ids):
        """
        Associate segmented pixels with projected LiDAR points and their corresponding 3D points.

        Args:
            projected_pixels: (N, 2) array of projected LiDAR pixel coordinates.
            valid_3d_ids: Indices of valid 3D points in the LiDAR cloud.

        Returns:
            associations: Dictionary mapping class labels to lists of associated 3D points.
        """
        associations = {}
        segment_pixels = []
        segmentation_results = self.last_seg_results  # list of (class_name, mask, conf)
        for class_label, seg_mask, conf in segmentation_results:
            # Find pixels belonging to the current class
            seg_indices = np.argwhere(seg_mask > 0)
            # Match segmented pixels to projected LiDAR pixels
            for seg_pixel in seg_indices:
                distances = np.linalg.norm(projected_pixels - seg_pixel, axis=1)
                closest_idx = np.argmin(distances)

                if distances[closest_idx] < 5.0:  # Threshold for pixel distance
                    point_id = valid_3d_ids[closest_idx]
                    point_3d = self.current_cloud[point_id]
                    segment_pixels.append(projected_pixels[closest_idx])

                    if class_label not in associations:
                        associations[class_label] = []
                    associations[class_label].append(point_3d)
        # print summary of associations
        for class_label, points in associations.items():
            print(f"Class '{class_label}' associated with {len(points)} 3D points")
        return associations, segment_pixels

    def publish_segmented_points(self, associations):
        # Build a colored PointCloud2 where each class has a unique color.
        if not associations:
            return
        from struct import pack, unpack
        from sensor_msgs.msg import PointField
        from std_msgs.msg import Header
        from sensor_msgs_py import point_cloud2 as pc2

        def pack_rgb_float(r, g, b):
            rgb_uint32 = (int(r) << 16) | (int(g) << 8) | int(b)
            return unpack("f", pack("I", rgb_uint32))[0]

        # Deterministic colors per class
        rng = np.random.default_rng(42)
        class_colors = {}
        for class_name in associations.keys():
            class_colors[class_name] = rng.integers(0, 255, size=3, dtype=np.uint8)

        points_out = []  # each element: (x, y, z, rgb_float)
        for class_name, points in associations.items():
            c = class_colors[class_name]
            rgb_f = pack_rgb_float(c[0], c[1], c[2])
            for p in points:
                points_out.append((float(p[0]), float(p[1]), float(p[2]), rgb_f))

        if not points_out:
            return

        header = Header()
        header.stamp = (
            self.current_msg_timestamp
            if hasattr(self, "current_msg_timestamp")
            else self.get_clock().now().to_msg()
        )
        header.frame_id = "base_link"  # Match the original KITTI point cloud frame

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="rgb", offset=12, datatype=PointField.FLOAT32, count=1
            ),  # rgb as packed float
        ]

        cloud_msg = pc2.create_cloud(header, fields, points_out)
        self.seg_points_pub.publish(cloud_msg)
        # Debug log
        self.get_logger().info(
            f"Published segmented cloud with {len(points_out)} points in frame '{header.frame_id}' across {len(associations)} classes"
        )

    def create_camera_info(self):
        msg = CameraInfo()

        # Set the camera matrix (K)
        msg.k = [
            9.037596e02,
            0.000000e00,
            6.957519e02,  # Row 1
            0.000000e00,
            9.019653e02,
            2.242509e02,  # Row 2
            0.000000e00,
            0.000000e00,
            1.000000e00,
        ]  # Row 3

        # Set the distortion parameters (D)
        msg.d = [
            -3.639558e-01,
            1.788651e-01,
            6.029694e-04,
            -3.922424e-04,
            -5.382460e-02,
        ]

        # Set the rectified camera matrix (P)
        msg.p = [
            7.215377e02,
            0.000000e00,
            6.095593e02,
            -3.395242e02,  # Row 1
            0.000000e00,
            7.215377e02,
            1.728540e02,
            2.199936e00,  # Row 2
            0.000000e00,
            0.000000e00,
            1.000000e00,
            2.729905e-03,
        ]  # Row 3

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

        # self.get_logger().info(
        #    f'Horizontal FOV: {fov_horizontal_degrees:.2f} degrees, '
        #    f'Vertical FOV: {fov_vertical_degrees:.2f} degrees'
        # )

        if self.angular_offset is None:
            return

        self.overlapping_sector = self.calculate_overlapping_sector(
            self.camera_h_fov, self.angular_offset
        )
        self.get_logger().info(f"Adjusted range: {self.overlapping_sector}")

    def map_update_callback(self):
        if self.odom is None:
            return

        # self.robot_x, self.robot_y, self.robot_yaw  = self.get_pose()
        pred_robot_x, pred_robot_y, pred_robot_yaw = None, None, None
        self.visualizer.update(
            self.landmarks,
            self.robot_x,
            self.robot_y,
            self.robot_yaw,
            pred_robot_x,
            pred_robot_y,
            pred_robot_yaw,
        )
        self.visualizer.update_loop_closure(self.loop_closure_landmarks)

    def view_features(
        self, cv_image, keypoints, projected_pixels, indices, segment_pixels
    ):
        green = []
        red = []
        alpha_projected = 0.45  # Transparency for projected points
        alpha_segmented = 0.35  # Transparency for segmented points

        # Draw keypoints within FOV in green
        for idx in indices:
            green.append(keypoints[idx])
        cv_image_with_keypoints = cv2.drawKeypoints(
            cv_image, green, None, color=(0, 255, 0), flags=0
        )

        # Draw keypoints outside FOV in red
        for i, kp in enumerate(keypoints):
            if i not in indices:
                red.append(keypoints[i])
        cv_image_with_keypoints = cv2.drawKeypoints(
            cv_image_with_keypoints, red, None, color=(0, 0, 255), flags=0
        )

        # Draw projected LiDAR points with transparency
        overlay = cv_image_with_keypoints.copy()
        for pt in projected_pixels:
            u, v = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(overlay, (u, v), 2, (255, 0, 255), -1, lineType=cv2.LINE_AA)
        cv_image_with_keypoints = cv2.addWeighted(
            overlay, alpha_projected, cv_image_with_keypoints, 1 - alpha_projected, 0
        )

        # Draw segmented points with transparency
        overlay = cv_image_with_keypoints.copy()
        if len(segment_pixels):
            for pt in segment_pixels:
                u, v = int(round(pt[0])), int(round(pt[1]))
                cv2.circle(overlay, (u, v), 3, (255, 255, 0), -1, lineType=cv2.LINE_AA)
        cv_image_with_keypoints = cv2.addWeighted(
            overlay, alpha_segmented, cv_image_with_keypoints, 1 - alpha_segmented, 0
        )

        # Resize for display consistency
        scale_factor = 0.5  # Downscale by half
        new_width = int(cv_image_with_keypoints.shape[1] * scale_factor)
        new_height = int(cv_image_with_keypoints.shape[0] * scale_factor)
        new_dimensions = (new_width, new_height)

        resized_image = cv2.resize(
            cv_image_with_keypoints, new_dimensions, interpolation=cv2.INTER_AREA
        )

        # Display the image
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
        return rot.as_euler("xyz")[2]

    def calculate_angular_offset(self, frame1, frame2, reference_frame):
        # Wait for the transform to be available
        while not self.tf_buffer.can_transform(
            reference_frame,
            frame1,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=1.0),
        ):
            self.get_logger().info(
                "Waiting for transform from {} to {}".format(reference_frame, frame1)
            )
            rclpy.spin_once(self, timeout_sec=0.1)

        while not self.tf_buffer.can_transform(
            reference_frame,
            frame2,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=1.0),
        ):
            self.get_logger().info(
                "Waiting for transform from {} to {}".format(reference_frame, frame2)
            )
            rclpy.spin_once(self, timeout_sec=0.1)

        try:
            # Get transforms to a common reference frame
            trans1 = self.tf_buffer.lookup_transform(
                reference_frame, frame1, rclpy.time.Time()
            )
            trans2 = self.tf_buffer.lookup_transform(
                reference_frame, frame2, rclpy.time.Time()
            )

            # Get yaw angles from quaternions
            yaw1 = self.quat_to_yaw(trans1.transform.rotation)
            yaw2 = self.quat_to_yaw(trans2.transform.rotation)

            # Calculate yaw (angular offset)
            yaw_offset = yaw2 - yaw1

            return yaw_offset  # Convert to degrees for readability
        except Exception as e:
            self.get_logger().error(f"Error calculating offset: {e}")
            return None

    def calculate_overlapping_sector(self, camera_fov, angular_offset):
        # Camera's angular range relative to its own forward direction
        camera_range = (-camera_fov / 2, camera_fov / 2)

        # Adjust by the angular offset to align with the range finder's frame
        overlapping_sector = (
            math.radians(camera_range[0] + angular_offset),
            math.radians(camera_range[1] + angular_offset),
        )

        return overlapping_sector

    def get_vertical_offset(self, source_frame, target_frame):
        try:
            # Wait for transform to be available
            while not self.tf_buffer.can_transform(
                target_frame, source_frame, rclpy.time.Time()
            ):
                self.get_logger().info("Waiting for transform")
                rclpy.spin_once(self, timeout_sec=0.1)

            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time()
            )
            return transform.transform.translation.z
        except Exception as e:
            self.get_logger().error("Error getting vertical offset: {}".format(e))
            return None

    def get_offset(self, source_frame, target_frame):
        try:
            # Wait for transform to be available
            while not self.tf_buffer.can_transform(
                target_frame, source_frame, rclpy.time.Time()
            ):
                self.get_logger().info("Waiting for transform")
                rclpy.spin_once(self, timeout_sec=0.1)

            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time()
            )

            # Extract the horizontal and vertical offsets
            offset_y = transform.transform.translation.y
            offset_z = transform.transform.translation.z

            return offset_y, offset_z

        except Exception as e:
            self.get_logger().error("Error getting offset: {}".format(e))
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


if __name__ == "__main__":
    main()
