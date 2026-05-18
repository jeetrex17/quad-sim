cmake_minimum_required(VERSION 3.8)
project(drone_sim)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(imu_sim src/imu_sim_node.cpp)
ament_target_dependencies(imu_sim rclcpp sensor_msgs)

install(TARGETS imu_sim DESTINATION lib/${PROJECT_NAME})

ament_package()
