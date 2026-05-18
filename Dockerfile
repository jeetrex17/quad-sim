FROM osrf/ros:humble-desktop

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-pybind11 \
    pybind11-dev \
    ros-humble-eigen3-cmake-module \
 && rm -rf /var/lib/apt/lists/*

RUN pip install "numpy<2" rerun-sdk
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install gymnasium stable-baselines3

WORKDIR /ros2_ws
