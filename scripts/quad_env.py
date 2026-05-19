#!/usr/bin/env python3
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_LIB_CANDIDATES = []

try:
    from ament_index_python.packages import get_package_prefix
    _LIB_CANDIDATES.append(Path(get_package_prefix("drone_sim")) / "lib" / "drone_sim")
except Exception:
    pass

for parent in _SCRIPT.parents:
    _LIB_CANDIDATES.append(parent / "install" / "drone_sim" / "lib" / "drone_sim")

for lib_dir in _LIB_CANDIDATES:
    if lib_dir.exists():
        sys.path.insert(0, str(lib_dir))
        break

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import quad_sim_cpp as qsc
except ModuleNotFoundError as e:
    tried = "\n".join(f"  - {path}" for path in _LIB_CANDIDATES)
    raise RuntimeError(
        "quad_sim_cpp not found. Build first: colcon build --packages-select drone_sim\n"
        f"Searched:\n{tried}"
    ) from e

W_HOVER  = qsc.W_HOVER
W_SCALE  = 85.0    # motor delta from hover (rad/s)
DT       = 0.005   # physics timestep (s)
MAX_STEPS = 600    # 3 s per episode
SUCCESS_TILT_DEG = 12.0
SUCCESS_OMEGA = 0.8
SUCCESS_Z_ERROR = 0.35
SUCCESS_HOLD_STEPS = int(round(0.20 / DT))

_OBS_HIGH = np.array([
    10, 10, 10,        # position (m)
     5,  5,  5,        # velocity (m/s)
     1,  1,  1,  1,   # quaternion (unit sphere)
    25, 25, 25,        # angular rate (rad/s)
], dtype=np.float32)


class QuadRecoveryEnv(gym.Env):
    """
    Task: recover from large random tilt and angular rate to stable hover.
    Observation: 13-dim QuadState (pos, vel, quat, omega).
    Action: 4 motor deltas in [-1, 1] mapped to W_HOVER +/- W_SCALE.
    """
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-_OBS_HIGH, _OBS_HIGH, dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self._state = qsc.QuadState()
        self._steps = 0
        self._stable_steps = 0
        self._prev_tilt = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        s = qsc.QuadState()
        max_tilt_deg = 85.0
        if options and "max_tilt_deg" in options:
            max_tilt_deg = float(options["max_tilt_deg"])

        # Mixed curriculum: keep easy recoveries in distribution but force the
        # policy to see aggressive attitudes often enough to learn recovery.
        mode = rng.random()
        if mode < 0.20:
            angle_deg = rng.uniform(2.0, min(20.0, max_tilt_deg))
        elif mode < 0.60:
            angle_deg = rng.uniform(20.0, min(55.0, max_tilt_deg))
        else:
            angle_deg = rng.uniform(55.0, max_tilt_deg)
        angle = np.deg2rad(angle_deg)

        axis  = rng.standard_normal(3)
        axis /= np.linalg.norm(axis) + 1e-8
        s.qw = float(np.cos(angle / 2))
        s.qx = float(axis[0] * np.sin(angle / 2))
        s.qy = float(axis[1] * np.sin(angle / 2))
        s.qz = float(axis[2] * np.sin(angle / 2))

        s.px, s.py, s.pz = 0.0, 0.0, 1.0
        s.vx, s.vy, s.vz = 0.0, 0.0, 0.0
        s.wx = float(rng.uniform(-3.0, 3.0))
        s.wy = float(rng.uniform(-3.0, 3.0))
        s.wz = float(rng.uniform(-1.0, 1.0))

        self._state = s
        self._steps = 0
        self._stable_steps = 0
        self._prev_tilt = self._tilt_rad(s)
        return self._obs(), {}

    def step(self, action):
        action  = np.clip(action, -1.0, 1.0)
        motors  = [float(np.clip(W_HOVER + a * W_SCALE, 50.0, 400.0)) for a in action]

        self._state = qsc.step(self._state, motors, DT)
        self._steps += 1

        obs        = self._obs()
        crashed    = self._crashed()
        success    = self._success()
        reward     = self._reward(action, success, crashed)
        terminated = crashed or success
        truncated  = self._steps >= MAX_STEPS
        return obs, reward, terminated, truncated, {
            "success": success,
            "crashed": crashed,
            "tilt_deg": np.rad2deg(self._tilt_rad(self._state)),
        }

    # ------------------------------------------------------------------
    def _obs(self):
        s = self._state
        return np.array([s.px, s.py, s.pz,
                         s.vx, s.vy, s.vz,
                         s.qw, s.qx, s.qy, s.qz,
                         s.wx, s.wy, s.wz], dtype=np.float32)

    def _tilt_rad(self, s):
        return 2.0 * np.arccos(np.clip(abs(s.qw), 0.0, 1.0))

    def _success(self):
        s = self._state
        tilt_deg = np.rad2deg(self._tilt_rad(s))
        omega = np.sqrt(s.wx**2 + s.wy**2 + s.wz**2)
        z_error = abs(s.pz - 1.0)

        if tilt_deg <= SUCCESS_TILT_DEG and omega <= SUCCESS_OMEGA and z_error <= SUCCESS_Z_ERROR:
            self._stable_steps += 1
        else:
            self._stable_steps = 0
        return self._stable_steps >= SUCCESS_HOLD_STEPS

    def _reward(self, action, success, crashed):
        s = self._state
        tilt = self._tilt_rad(s)
        tilt_progress = self._prev_tilt - tilt
        self._prev_tilt = tilt

        omega_sq = s.wx**2 + s.wy**2 + s.wz**2
        vel_sq = s.vx**2 + s.vy**2 + s.vz**2
        z_error = s.pz - 1.0
        action_sq = float(np.mean(np.square(action)))

        reward = (
            1.5
            + 8.0 * tilt_progress
            - 4.0 * (tilt / np.pi)
            - 0.08 * omega_sq
            - 0.04 * vel_sq
            - 0.8 * z_error * z_error
            - 0.015 * action_sq
        )
        if self._stable_steps > 0:
            reward += 0.10 * self._stable_steps
        if success:
            reward += 50.0
        if crashed:
            reward -= 50.0
        return float(reward)

    def _crashed(self):
        s = self._state
        if s.pz < -0.1 or s.pz > 8.0:
            return True
        if abs(s.px) > 6.0 or abs(s.py) > 6.0:
            return True
        if self._tilt_rad(s) > np.deg2rad(110.0):
            return True
        return False
