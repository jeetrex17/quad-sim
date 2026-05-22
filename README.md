# quad-sim

![quad-sim Rerun demo](docs/demo.gif)

6-DOF quadrotor simulator with cascaded PID control, MEKF attitude estimation, and a PPO-trained attitude recovery policy. Physics core in C++17, exposed to Python via pybind11 for RL training and offline benchmarking.

## Stack

| Layer | Detail |
|-------|--------|
| Physics | C++17, RK4 integrator, quaternion kinematics |
| Estimation | MEKF, 6-state error state, Joseph-form covariance update |
| Control | Cascaded PID, X-config motor mixing |
| RL | Gymnasium env, SB3 PPO, pybind11 C++ physics backend |
| Viz | rerun.io, drone model, flight cage, gates, recovery wall, live trail, target, and telemetry plots |
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

## Benchmarks

Reproduce the benchmark table:

```bash
ros2 run drone_sim benchmark.py \
  --seed 7 \
  --duration 28 \
  --rl-episodes 200 \
  --speed-steps 100000 \
  --markdown docs/benchmark.md \
  --json docs/benchmark.json
```

Seed `7`, 28 s PID/MEKF trajectory, 200 randomized RL recovery trials.

| Area | Metric | Result |
|------|--------|--------|
| PID trajectory | RMS position error | 0.821 m |
| PID trajectory | Mean position error | 0.717 m |
| PID trajectory | Max position error | 1.260 m |
| PID trajectory | Final landing error | 0.055 m |
| MEKF attitude | Attitude RMSE | 8.34 deg |
| MEKF attitude | 95th percentile attitude error | 11.50 deg |
| MEKF attitude | Roll / pitch / yaw RMSE | 0.08 / 0.08 / 8.34 deg |
| RL recovery | Success rate | 200/200 (100.0%) |
| RL recovery | Crash rate | 0/200 (0.0%) |
| RL recovery | Mean recovery time | 0.47 s |
| RL recovery | Max recovered initial tilt | 84.9 deg |
| Physics core | C++ RK4 throughput | 265,022 steps/s |

MEKF roll/pitch error is low; yaw dominates full attitude RMSE because this estimator fuses gyro + accelerometer only, so yaw is not gravity-observable.

Machine-readable results are in `docs/benchmark.json`.

## Components

**`include/drone_sim/quad_dynamics.hpp`** - pure C++ physics, no ROS deps. DJI F450 constants, RK4 integrator, quaternion kinematics. Single source of truth shared by the ROS node and the Python RL env via pybind11.

**`drone_dynamics_node`** - steps physics at 200 Hz, publishes ground-truth odometry and a simulated ADIS16470 IMU (ARW 4.65e-4 rad/s/rtHz, random walk + bias drift).

**`mekf_node`** - multiplicative EKF. Predicts via quaternion kinematics + gyro, corrects against accelerometer gravity reference. Skips correction when `|a| - g > 2 m/s^2` (maneuver detection).

**`pid_controller.py`** - cascaded outer loop (position -> desired thrust + attitude) and inner loop (attitude error -> torques). Motor commands from inverse X-config allocation matrix.

**`quad_env.py`** - Gymnasium environment. Resets to randomized tilt up to 85 deg with angular rates up to 3 rad/s. Reward penalizes tilt, omega, velocity, altitude error, and excess action. Terminates on crash or out-of-bounds.

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
