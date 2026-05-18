FROM osrf/ros:humble-desktop

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-pybind11 \
    pybind11-dev \
    ros-humble-eigen3-cmake-module \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/quad-sim-requirements.txt
RUN pip install -r /tmp/quad-sim-requirements.txt

WORKDIR /ros2_ws
