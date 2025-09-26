# simple_lidar_odometry_cpp

C++ port of the Python `simple_lidar_odometry` node. Performs frame-to-frame point-to-plane ICP using Open3D, accumulates an odometry transform, and publishes `nav_msgs/Odometry` on `pointcloud/odom`.

## Build
```bash
colcon build --packages-select simple_lidar_odometry_cpp
source install/setup.bash
```

## Run
```bash
ros2 run simple_lidar_odometry_cpp lidar_odometry_node
```

Ensure the input point cloud topic `/kitti/point_cloud` is available (e.g. via rosbag play).

## Loop Detection
A loop (rosbag restart) is detected if message time stamp jumps backward or forward by more than 5 seconds; internal odometry state is reset.

## Dependencies
- rclcpp
- sensor_msgs
- nav_msgs
- geometry_msgs
- Open3D (system install providing CMake config)
- Eigen3

Adjust voxel size, thresholds, and loop detection threshold in `src/lidar_odometry_node.cpp` as needed.
