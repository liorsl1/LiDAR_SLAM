cmake_minimum_required(VERSION 3.8)
project(simple_lidar_odometry_cpp)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
# Open3D provides a CMake config
find_package(Open3D REQUIRED)

add_executable(lidar_odometry_node src/lidar_odometry_node.cpp)
ament_target_dependencies(lidar_odometry_node rclcpp sensor_msgs nav_msgs geometry_msgs)

target_link_libraries(lidar_odometry_node Open3D::Open3D Eigen3::Eigen)

install(TARGETS lidar_odometry_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
