# LiDAR_SLAM

Lightweight educational SLAM sandbox with two focused modules:

1. LiDAR Odometry (point‑to‑plane ICP)
2. Visual–LiDAR Mapping ,Loop Closure, and Graph Optimization (feature fusion + graph optimization)

---
## 1. LiDAR Odometry Module
Script: `simple_lidar_icp_odometry.py`

Odometry estimation from frame‑to‑frame Open3D point‑to‑plane ICP on `/kitti/point_cloud`, integrates relative transforms, publishes `nav_msgs/Odometry` on `pointcloud/odom`, resets on large time jumps.

Showcase image (RViz: purple arrows = pose chain):
![LiDAR Odometry Visualization](readme_files/img.png)

---
## 2. Mapping & Loop Closure Module
Folder: `mapping_optimize/`

Clusters ORB keypoints (DBSCAN), fuses with LiDAR ranges to create 2D landmarks (spatial hash + descriptor gating), builds a pose/landmark graph, injects Gaussian noise, then optimizes with g2o (Gauss‑Newton / Levenberg / Dogleg) to reduce drift and tighten landmark dispersion.

Showcase animation:
![Mapping & Optimization](readme_files/mapping.gif)

See `mapping_optimize/README.md` for solver comparison thumbnails and details.

---
## Quick Install (core Python deps)
```bash
pip install numpy open3d scipy opencv-python scikit-learn plotly
```
Source your ROS 2 (Humble) environment for message types and `rclpy`.

---
## Run Examples
LiDAR odometry:
```bash
python simple_lidar_icp_odometry.py
```
Mapping / loop closure visualization (after topics available):
```bash
python mapping_optimize/vis_mapping_closure.py
```

---
## Video (inline if supported)
<video src="readme_files/vid.mp4" controls loop muted playsinline width="640"></video>

[View Video](readme_files/vid.mp4)
