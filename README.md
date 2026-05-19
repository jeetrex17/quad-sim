# quad-sim

6-DOF quadrotor simulator with cascaded PID control, MEKF attitude estimation, and a PPO-trained attitude recovery policy. Physics core in C++17, exposed to Python via pybind11 for RL training at ~1000 steps/sec on CPU.

## Stack

| Layer | Detail |
|-------|--------|
| Physics | C++17, RK4 integrator, quaternion kinematics |
| Estimation | MEKF, 6-state error state, Joseph-form covariance update |
| Control | Cascaded PID, X-config motor mixing |
| RL | Gymnasium env, SB3 PPO, pybind11 C++ physics backend |
| Viz | rerun.io, procedural flight lab, drone model, path, targets, and live plots |
| Middleware | ROS 2 Humble |

## Build

```bash
docker run -it osrf/ros:humble-desktop bash

# inside container
mkdir -p /ros2_ws/src
git clone https://github.com/jeetrex17/quad-sim /ros2_ws/src/quad-sim
cd /ros2_ws

apt-get install -y python3-pybind11 pybind11-dev ros-humble-eigen3-cmake-module
pip install -r src/quad-sim/requirements.txt

colcon build --packages-select drone_sim
source install/setup.bash
```

## Run

PID waypoint following + live visualization:
```bash
ros2 launch drone_sim sim.launch.py
```

RL attitude recovery evaluation:
```bash
ros2 launch drone_sim rl_eval.launch.py
```
If `scripts/quad_recovery_policy.zip` is not present, run the training command below first.

Trigger a recovery event (separate terminal):
```bash
ros2 topic pub --once /drone/motor_speeds std_msgs/msg/Float64MultiArray \
  "{data: [250.0, 150.0, 250.0, 150.0]}"
```

Open the rerun viewer on the Mac host: `rerun`

## Components

**`include/drone_sim/quad_dynamics.hpp`** - pure C++ physics, no ROS deps. DJI F450 constants, RK4 integrator, quaternion kinematics. Single source of truth shared by the ROS node and the Python RL env via pybind11.

**`drone_dynamics_node`** - steps physics at 200 Hz, publishes ground-truth odometry and a simulated ADIS16470 IMU (ARW 4.65e-4 rad/s/rtHz, random walk + bias drift).

**`mekf_node`** - multiplicative EKF. Predicts via quaternion kinematics + gyro, corrects against accelerometer gravity reference. Skips correction when `|a| - g > 2 m/s^2` (maneuver detection).

**`pid_controller.py`** - cascaded outer loop (position -> desired thrust + attitude) and inner loop (attitude error -> torques). Motor commands from inverse X-config allocation matrix.

**`quad_env.py`** - Gymnasium environment. Resets to random tilt 17-60 deg with angular rates up to 3 rad/s. Reward penalizes tilt, omega, and altitude error. Terminates on tilt > 90 deg or out-of-bounds.

**`rl_recovery_node.py`** - 200 Hz node. Monitors MEKF quaternion. Engages PPO policy when tilt > 45 deg, disengages at < 20 deg. Hysteresis prevents chattering at the boundary.

## Training

```bash
cd /ros2_ws
python3 scripts/train_ppo.py --timesteps 2000000 --n-envs 8
```

Evaluate the trained recovery policy:
```bash
ros2 run drone_sim evaluate_policy.py --episodes 200
```

| Metric | Iter 1 | Iter 245 |
|--------|--------|----------|
| ep_len_mean | 87 | 391 |
| per-step reward | -0.87 | -0.12 |
| explained_variance | 0.03 | 0.98 |
| value_loss | 211 | 1.85 |

2M PPO steps. Recovery from high initial tilt can be evaluated with randomized trials.
