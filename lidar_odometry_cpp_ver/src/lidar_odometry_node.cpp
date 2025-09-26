#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <open3d/Open3D.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <memory>
#include <vector>
#include <string>
#include <cmath>

// C++ port of the Python LidarOdometry node.
// Functionality:
//  * Subscribe to a PointCloud2 topic (/kitti/point_cloud)
//  * Convert to Open3D point cloud, voxel downsample
//  * Perform point-to-plane ICP between consecutive frames
//  * Accumulate transform into an odometry matrix
//  * Publish incremental odometry as nav_msgs/Odometry
//  * Detect rosbag loop restarts by timestamp jumps and reset state

class LidarOdometryNode : public rclcpp::Node {
public:
  LidarOdometryNode() : Node("lidar_odometry_node"),
                        odometry_(Eigen::Matrix4d::Identity()),
                        transformation_(Eigen::Matrix4d::Identity()),
                        last_timestamp_(-1.0),
                        loop_detection_threshold_(5.0),
                        loop_just_detected_(false) {

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("pointcloud/odom", 10);

    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/kitti/point_cloud", 10,
        std::bind(&LidarOdometryNode::cloudCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "LidarOdometryNode (C++) initialized.");
  }

private:
  // Publishers and subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

  // State clouds
  std::shared_ptr<open3d::geometry::PointCloud> prev_cloud_;

  // Transformation state
  Eigen::Matrix4d odometry_;
  Eigen::Matrix4d transformation_;

  // Loop detection
  double last_timestamp_;
  double loop_detection_threshold_;
  bool loop_just_detected_;

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (detectNewLoop(msg)) {
      resetVariables();
    }

    auto cloud = pointCloud2ToOpen3D(*msg);
    if (!prev_cloud_) {
      prev_cloud_ = cloud;
      return; // Need a previous frame for ICP
    }

    double inlier_rmse = 0.0;
    performPointToPlaneICP(cloud, prev_cloud_, transformation_, inlier_rmse);

    // Accumulate global odometry
    odometry_ = odometry_ * transformation_;

    // Extract rotation and translation
    Eigen::Matrix3d R = odometry_.block<3,3>(0,0);
    Eigen::Vector3d T = odometry_.block<3,1>(0,3);

    Eigen::Vector3d euler_deg = rotationMatrixToEulerXYZDegrees(R);
    RCLCPP_INFO(this->get_logger(), "LiDAR Odom: x: %.3f, y: %.3f, yaw: %.3f", T.x(), T.y(), euler_deg.z());

    publishOdometry(T.x(), T.y(), T.z(), R);

    prev_cloud_ = cloud;
  }

  bool detectNewLoop(const sensor_msgs::msg::PointCloud2::SharedPtr & msg) {
    double current_time = static_cast<double>(msg->header.stamp.sec) +
                          static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    if (last_timestamp_ >= 0.0) {
      double dt = current_time - last_timestamp_;
      if ((dt < -loop_detection_threshold_ || dt > loop_detection_threshold_) && !loop_just_detected_) {
        RCLCPP_INFO(this->get_logger(), "Loop detected (dt=%.2f) resetting state", dt);
        loop_just_detected_ = true;
        last_timestamp_ = current_time; // prevent repeat triggers
        return true;
      }
      // Normal forward progression resets guard for next loop detection
      if (dt >= 0.0 && dt <= loop_detection_threshold_) {
        loop_just_detected_ = false;
      }
    }
    last_timestamp_ = current_time;
    return false;
  }

  void resetVariables() {
    prev_cloud_.reset();
    odometry_.setIdentity();
    transformation_.setIdentity();
    RCLCPP_INFO(this->get_logger(), "State reset after loop detection");
  }

  std::shared_ptr<open3d::geometry::PointCloud> pointCloud2ToOpen3D(const sensor_msgs::msg::PointCloud2 & ros_cloud) {
    // Gather points, replicating Python's field order (y,x,z)
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(ros_cloud.width * ros_cloud.height);

    sensor_msgs::PointCloud2ConstIterator<float> it_x(ros_cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(ros_cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(ros_cloud, "z");

    for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z) {
      float x = *it_x; float y = *it_y; float z = *it_z;
      if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
        pts.emplace_back(Eigen::Vector3d(y, x, z));
      }
    }

    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    cloud->points_.reserve(pts.size());
    for (auto & p : pts) cloud->points_.push_back(p);

    // Downsample (voxel size = 1.0 similar to voxel_down_sample(voxel_size=1))
    auto down = cloud->VoxelDownSample(1.0);
    return down;
  }

  void performPointToPlaneICP(const std::shared_ptr<open3d::geometry::PointCloud> & source,
                              const std::shared_ptr<open3d::geometry::PointCloud> & target,
                              Eigen::Matrix4d & out_transform,
                              double & out_inlier_rmse) {
    source->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(1.0, 30));
    target->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(1.0, 30));

    double threshold = 1.0;
    Eigen::Matrix4d init = out_transform; // prior

    auto result = open3d::pipelines::registration::RegistrationICP(
      *source,
      *target,
      threshold,
      init,
      open3d::pipelines::registration::TransformationEstimationPointToPlane()
    );

    out_transform = result.transformation_;
    out_inlier_rmse = result.inlier_rmse_;
  }

  static Eigen::Vector3d rotationMatrixToEulerXYZDegrees(const Eigen::Matrix3d & R) {
    // Eigen's eulerAngles(0,1,2) -> XYZ intrinsic (matches Python scipy 'xyz')
    Eigen::Vector3d euler = R.eulerAngles(0,1,2);
    euler *= (180.0 / M_PI);
    return euler;
  }

  void publishOdometry(double x, double y, double z, const Eigen::Matrix3d & R) {
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = this->get_clock()->now();
    odom.header.frame_id = "odom";
    odom.child_frame_id = ""; // matches Python

    // Match Python sign inversion on x
    odom.pose.pose.position.x = -x;
    odom.pose.pose.position.y = y;
    odom.pose.pose.position.z = z;

    // Python used quaternion from transposed R
    Eigen::Quaterniond q(R.transpose());
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    odom_pub_->publish(odom);
  }
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LidarOdometryNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
