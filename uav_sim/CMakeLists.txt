cmake_minimum_required(VERSION 3.5)
project(uav_sim)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)

# vehicle_odometry_recorder
add_executable(vehicle_odometry_recorder src/vehicle_odometry_recorder.cpp)
ament_target_dependencies(vehicle_odometry_recorder rclcpp px4_msgs)
install(TARGETS vehicle_odometry_recorder DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

# Export
ament_export_dependencies(ament_cmake rclcpp px4_msgs)
ament_package()
