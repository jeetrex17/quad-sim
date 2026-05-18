cmake_minimum_required(VERSION 3.8)
project(drone_sim)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 QUIET)

include_directories(include)

add_executable(imu_sim src/imu_sim_node.cpp)
ament_target_dependencies(imu_sim rclcpp sensor_msgs)

add_executable(drone_dynamics src/drone_dynamics_node.cpp)
ament_target_dependencies(drone_dynamics rclcpp sensor_msgs nav_msgs std_msgs)

add_executable(mekf src/mekf_node.cpp)
target_include_directories(mekf PRIVATE ${EIGEN3_INCLUDE_DIRS})
ament_target_dependencies(mekf rclcpp sensor_msgs geometry_msgs)

if(pybind11_FOUND)
  pybind11_add_module(quad_sim_cpp src/bindings.cpp)
  target_include_directories(quad_sim_cpp PRIVATE include)
  install(TARGETS quad_sim_cpp DESTINATION lib/${PROJECT_NAME})
  message(STATUS "pybind11 found — building Python extension quad_sim_cpp")
else()
  message(WARNING "pybind11 not found — skipping Python bindings")
endif()

install(TARGETS imu_sim drone_dynamics mekf DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

install(PROGRAMS
  scripts/pid_controller.py
  scripts/rerun_viz.py
  scripts/rl_recovery_node.py
  scripts/teleop_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  scripts/quad_recovery_policy.zip
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
