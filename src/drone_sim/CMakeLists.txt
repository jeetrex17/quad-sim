cmake_minimum_required(VERSION 3.8)
project(drone_sim)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(imu_sim src/imu_sim_node.cpp)
ament_target_dependencies(imu_sim rclcpp sensor_msgs)

add_executable(drone_dynamics src/drone_dynamics_node.cpp)
ament_target_dependencies(drone_dynamics rclcpp sensor_msgs nav_msgs std_msgs)

add_executable(mekf src/mekf_node.cpp)
target_include_directories(mekf PRIVATE ${EIGEN3_INCLUDE_DIRS})
ament_target_dependencies(mekf rclcpp sensor_msgs geometry_msgs)

install(TARGETS imu_sim drone_dynamics mekf DESTINATION lib/${PROJECT_NAME})

ament_package()
