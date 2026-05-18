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
W_SCALE  = 50.0    # motor delta from hover (rad/s)
DT       = 0.005   # physics timestep (s)
MAX_STEPS = 600    # 3 s per episode

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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        s = qsc.QuadState()

        # Curriculum: 30% easy (5-25 deg) so policy sees near-hover states,
        # 70% hard (25-80 deg) for robustness
        if rng.random() < 0.3:
            angle = rng.uniform(0.09, 0.44)   # 5-25 deg
        else:
            angle = rng.uniform(0.44, 1.40)   # 25-80 deg

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
        return self._obs(), {}

    def step(self, action):
        action  = np.clip(action, -1.0, 1.0)
        motors  = [float(np.clip(W_HOVER + a * W_SCALE, 50.0, 400.0)) for a in action]

        self._state = qsc.step(self._state, motors, DT)
        self._steps += 1

        obs        = self._obs()
        reward     = self._reward()
        terminated = self._is_done()
        truncated  = self._steps >= MAX_STEPS
        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def _obs(self):
        s = self._state
        return np.array([s.px, s.py, s.pz,
                         s.vx, s.vy, s.vz,
                         s.qw, s.qx, s.qy, s.qz,
                         s.wx, s.wy, s.wz], dtype=np.float32)

    def _reward(self):
        s = self._state
        tilt       = 1.0 - s.qw * s.qw   # 0 upright, 0.5 at 90 deg
        omega_sq   = s.wx**2 + s.wy**2 + s.wz**2
        vel_sq     = s.vx**2 + s.vy**2 + s.vz**2
        height_err = (s.pz - 1.0)**2

        # Explicit goal-state bonus: nearly upright + calm = clear target
        # tilt < 0.05 is ~13 deg, omega_sq < 0.5 is ~0.7 rad/s per axis
        stability_bonus = 2.0 if (tilt < 0.05 and omega_sq < 0.5) else 0.0

        return float(
             1.0                    # alive bonus
           + stability_bonus        # goal state reward
           - 2.5  * tilt           # stronger tilt penalty
           - 0.1  * omega_sq       # damp angular rates
           - 0.05 * vel_sq         # damp translation
           - 0.2  * height_err     # hold z = 1 m
        )

    def _is_done(self):
        s = self._state
        if s.pz < -0.1 or s.pz > 8.0:
            return True
        if abs(s.px) > 6.0 or abs(s.py) > 6.0:
            return True
        # qw^2 < 0.5 means tilt > 90 deg, unrecoverable for this task
        if s.qw * s.qw < 0.5:
            return True
        return False
