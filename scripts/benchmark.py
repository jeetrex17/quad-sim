#!/usr/bin/env python3
import argparse
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

from quad_env import DT, QuadRecoveryEnv, qsc


GRAVITY = qsc.GRAVITY
MASS = qsc.MASS
KT = qsc.KT
KQ = qsc.KQ
L = qsc.L

GYRO_ARW = 4.65e-4
ACCEL_VRW = 1.03e-3
GYRO_BWALK = 1.0e-6
ACCEL_BWALK = 5.0e-5


@dataclass
class PidMetrics:
    duration_s: float
    rms_position_error_m: float
    mean_position_error_m: float
    max_position_error_m: float
    altitude_rmse_m: float
    final_position_error_m: float
    max_motor_speed_rad_s: float


@dataclass
class MekfMetrics:
    attitude_rmse_deg: float
    attitude_p95_deg: float
    roll_rmse_deg: float
    pitch_rmse_deg: float
    yaw_rmse_deg: float
    accel_updates: int
    accel_skips: int


@dataclass
class RlMetrics:
    episodes: int
    successes: int
    crashes: int
    success_rate_pct: float
    crash_rate_pct: float
    mean_recovery_time_s: float
    max_recovered_initial_tilt_deg: float
    mean_final_tilt_deg: float
    mean_z_error_m: float
    mean_omega_norm_rad_s: float


@dataclass
class PhysicsMetrics:
    steps: int
    elapsed_s: float
    steps_per_second: float


class PID:
    def __init__(self, kp, ki, kd, i_limit=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_limit = i_limit
        self._i = 0.0

    def step(self, error, rate, dt):
        self._i = float(np.clip(self._i + error * dt, -self.i_limit, self.i_limit))
        return self.kp * error + self.ki * self._i - self.kd * rate


class Mekf:
    def __init__(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.b = np.zeros(3, dtype=np.float64)
        self.P = np.zeros((6, 6), dtype=np.float64)
        self.P[0:3, 0:3] = 0.1 * np.eye(3)
        self.P[3:6, 3:6] = 1e-4 * np.eye(3)
        self.Q = np.zeros((6, 6), dtype=np.float64)
        self.Q[0:3, 0:3] = (GYRO_ARW * GYRO_ARW / DT) * np.eye(3)
        self.Q[3:6, 3:6] = (GYRO_BWALK * GYRO_BWALK) * np.eye(3)
        self.R_accel = (ACCEL_VRW * ACCEL_VRW / DT) * np.eye(3)
        self.accel_updates = 0
        self.accel_skips = 0

    def tick(self, omega_raw, accel):
        self._predict(omega_raw)
        self._correct(accel)
        return self.q.copy()

    def _predict(self, omega_raw):
        w = omega_raw - self.b
        self.q = qintegrate(self.q, w, DT)

        phi = np.eye(6)
        phi[0:3, 0:3] -= skew(w) * DT
        phi[0:3, 3:6] = -np.eye(3) * DT
        self.P = phi @ self.P @ phi.T + self.Q

    def _correct(self, accel):
        if abs(np.linalg.norm(accel) - GRAVITY) > 2.0:
            self.accel_skips += 1
            return

        g_body = qR(self.q).T @ np.array([0.0, 0.0, GRAVITY])
        H = np.zeros((3, 6), dtype=np.float64)
        H[0:3, 0:3] = skew(g_body)
        y = accel - g_body
        S = H @ self.P @ H.T + self.R_accel
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        dtheta = dx[0:3]
        dq = np.array([1.0, dtheta[0] / 2.0, dtheta[1] / 2.0, dtheta[2] / 2.0])
        self.q = qnorm(qmul(self.q, dq))
        self.b += dx[3:6]

        I = np.eye(6)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R_accel @ K.T
        self.accel_updates += 1


def make_state():
    s = qsc.QuadState()
    s.px, s.py, s.pz = 0.0, 0.0, 1.0
    s.vx, s.vy, s.vz = 0.0, 0.0, 0.0
    s.qw, s.qx, s.qy, s.qz = 1.0, 0.0, 0.0, 0.0
    s.wx, s.wy, s.wz = 0.0, 0.0, 0.0
    return s


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def qmul(p, q):
    return np.array([
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
        p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
    ], dtype=np.float64)


def qnorm(q):
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def qR(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def qintegrate(q, omega, dt):
    angle = np.linalg.norm(omega) * dt
    if angle < 1e-10:
        dq = np.array([1.0, omega[0] * dt / 2.0, omega[1] * dt / 2.0, omega[2] * dt / 2.0])
    else:
        axis = omega / np.linalg.norm(omega)
        s = math.sin(angle / 2.0)
        dq = np.array([math.cos(angle / 2.0), s * axis[0], s * axis[1], s * axis[2]])
    return qnorm(qmul(q, dq))


def quat_to_euler(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(clamp(2 * (w * y - z * x), -1.0, 1.0))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def angle_diff_deg(a, b):
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return np.degrees(d)


def quat_error_deg(q_est, q_true):
    dot = abs(float(np.dot(qnorm(q_est), qnorm(q_true))))
    return 2.0 * math.degrees(math.acos(clamp(dot, 0.0, 1.0)))


def waypoint_at(t):
    wps = [
        (0.0, 0.0, 0.0, 1.0),
        (5.0, 0.0, 0.0, 3.0),
        (9.0, 2.0, 0.0, 3.0),
        (13.0, 2.0, 2.0, 3.0),
        (17.0, 0.0, 2.0, 3.0),
        (21.0, 0.0, 0.0, 3.0),
        (25.0, 0.0, 0.0, 1.0),
    ]
    if t >= wps[-1][0]:
        return np.array(wps[-1][1:], dtype=np.float64)
    for i in range(len(wps) - 1):
        t0, x0, y0, z0 = wps[i]
        t1, x1, y1, z1 = wps[i + 1]
        if t0 <= t < t1:
            a = (t - t0) / (t1 - t0)
            return np.array([x0 + a * (x1 - x0), y0 + a * (y1 - y0), z0 + a * (z1 - z0)], dtype=np.float64)
    return np.array(wps[0][1:], dtype=np.float64)


def motor_mixing(F, tau_x, tau_y, tau_z):
    w_sq = [
        F / (4 * KT) - tau_y / (2 * KT * L) - tau_z / (4 * KQ),
        F / (4 * KT) + tau_x / (2 * KT * L) + tau_z / (4 * KQ),
        F / (4 * KT) + tau_y / (2 * KT * L) - tau_z / (4 * KQ),
        F / (4 * KT) - tau_x / (2 * KT * L) + tau_z / (4 * KQ),
    ]
    return [math.sqrt(max(0.0, w)) for w in w_sq]


def pid_motors(state, target, pids):
    q = np.array([state.qw, state.qx, state.qy, state.qz])
    roll, pitch, yaw = quat_to_euler(q)

    F = MASS * GRAVITY + pids["z"].step(target[2] - state.pz, state.vz, DT)
    F = clamp(F, 0.0, 2.5 * MASS * GRAVITY)

    ax_d = pids["x"].step(target[0] - state.px, state.vx, DT)
    ay_d = pids["y"].step(target[1] - state.py, state.vy, DT)
    pitch_d = clamp(ax_d / GRAVITY, -0.35, 0.35)
    roll_d = clamp(-ay_d / GRAVITY, -0.35, 0.35)

    tau_x = pids["roll"].step(roll_d - roll, state.wx, DT)
    tau_y = pids["pitch"].step(pitch_d - pitch, state.wy, DT)
    tau_z = pids["yaw"].step(0.0 - yaw, state.wz, DT)
    return motor_mixing(F, tau_x, tau_y, tau_z)


def run_pid_mekf(duration_s, seed):
    rng = np.random.default_rng(seed)
    state = make_state()
    mekf = Mekf()
    pids = {
        "z": PID(6.0, 0.3, 4.0),
        "x": PID(0.5, 0.0, 1.5),
        "y": PID(0.5, 0.0, 1.5),
        "roll": PID(6.0, 0.0, 1.0, i_limit=0.5),
        "pitch": PID(6.0, 0.0, 1.0, i_limit=0.5),
        "yaw": PID(2.0, 0.0, 0.5, i_limit=0.5),
    }

    pos_errors = []
    altitude_errors = []
    motor_max = 0.0
    attitude_errors = []
    rpy_errors = []

    gyro_bias = np.zeros(3)
    accel_bias = np.zeros(3)
    gyro_noise_std = GYRO_ARW * math.sqrt(1.0 / DT)
    accel_noise_std = ACCEL_VRW * math.sqrt(1.0 / DT)
    steps = int(round(duration_s / DT))

    for step in range(steps):
        t = step * DT
        target = waypoint_at(t)
        motors = pid_motors(state, target, pids)
        motor_max = max(motor_max, max(motors))
        state = qsc.step(state, motors, DT)

        if state.pz < 0.0:
            state.pz = 0.0
            state.vx = state.vy = state.vz = 0.0
        if state.px > 3.0:
            state.px = 3.0
            state.vx = state.vy = state.vz = 0.0

        pos = np.array([state.px, state.py, state.pz], dtype=np.float64)
        err = pos - target
        pos_errors.append(float(np.linalg.norm(err)))
        altitude_errors.append(float(err[2]))

        gyro_bias += rng.normal(0.0, GYRO_BWALK, 3)
        accel_bias += rng.normal(0.0, ACCEL_BWALK, 3)
        q_true = np.array([state.qw, state.qx, state.qy, state.qz], dtype=np.float64)
        gyro = np.array([state.wx, state.wy, state.wz]) + gyro_bias + rng.normal(0.0, gyro_noise_std, 3)
        accel = qR(q_true).T @ np.array([0.0, 0.0, GRAVITY])
        accel = accel + accel_bias + rng.normal(0.0, accel_noise_std, 3)
        q_est = mekf.tick(gyro, accel)

        if t >= 0.5:
            attitude_errors.append(quat_error_deg(q_est, q_true))
            rpy_errors.append(angle_diff_deg(quat_to_euler(q_est), quat_to_euler(q_true)))

    final_target = waypoint_at(duration_s)
    final_pos = np.array([state.px, state.py, state.pz], dtype=np.float64)
    pos_errors = np.asarray(pos_errors)
    altitude_errors = np.asarray(altitude_errors)
    attitude_errors = np.asarray(attitude_errors)
    rpy_errors = np.asarray(rpy_errors)

    pid = PidMetrics(
        duration_s=duration_s,
        rms_position_error_m=float(np.sqrt(np.mean(pos_errors ** 2))),
        mean_position_error_m=float(np.mean(pos_errors)),
        max_position_error_m=float(np.max(pos_errors)),
        altitude_rmse_m=float(np.sqrt(np.mean(altitude_errors ** 2))),
        final_position_error_m=float(np.linalg.norm(final_pos - final_target)),
        max_motor_speed_rad_s=float(motor_max),
    )
    mekf_metrics = MekfMetrics(
        attitude_rmse_deg=float(np.sqrt(np.mean(attitude_errors ** 2))),
        attitude_p95_deg=float(np.percentile(attitude_errors, 95)),
        roll_rmse_deg=float(np.sqrt(np.mean(rpy_errors[:, 0] ** 2))),
        pitch_rmse_deg=float(np.sqrt(np.mean(rpy_errors[:, 1] ** 2))),
        yaw_rmse_deg=float(np.sqrt(np.mean(rpy_errors[:, 2] ** 2))),
        accel_updates=mekf.accel_updates,
        accel_skips=mekf.accel_skips,
    )
    return pid, mekf_metrics


def mean_or_nan(values):
    values = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(values)) if values else math.nan


def run_rl(episodes, seed, policy, stochastic):
    from stable_baselines3 import PPO
    from evaluate_policy import _resolve_policy, _run_episode

    args = SimpleNamespace(
        stochastic=stochastic,
        success_tilt_deg=12.0,
        success_omega=0.8,
        success_z_error=0.35,
        hold_s=0.20,
    )
    env = QuadRecoveryEnv()
    policy_path = _resolve_policy(policy)
    model = PPO.load(policy_path)
    results = [_run_episode(model, env, episode, seed + episode, args) for episode in range(episodes)]

    successes = [r for r in results if r.success]
    crashes = [r for r in results if r.crashed]
    return RlMetrics(
        episodes=episodes,
        successes=len(successes),
        crashes=len(crashes),
        success_rate_pct=100.0 * len(successes) / episodes if episodes else 0.0,
        crash_rate_pct=100.0 * len(crashes) / episodes if episodes else 0.0,
        mean_recovery_time_s=mean_or_nan(r.recovery_time_s for r in successes),
        max_recovered_initial_tilt_deg=float(max((r.initial_tilt_deg for r in successes), default=math.nan)),
        mean_final_tilt_deg=mean_or_nan(r.final_tilt_deg for r in results),
        mean_z_error_m=mean_or_nan(r.final_z_error_m for r in results),
        mean_omega_norm_rad_s=mean_or_nan(r.final_omega_norm_rad_s for r in results),
    )


def run_physics_speed(steps):
    state = make_state()
    motors = [qsc.W_HOVER] * 4
    start = time.perf_counter()
    for _ in range(steps):
        state = qsc.step(state, motors, DT)
    elapsed = time.perf_counter() - start
    return PhysicsMetrics(steps=steps, elapsed_s=elapsed, steps_per_second=steps / elapsed)


def fmt_float(value, unit="", digits=3):
    if value is None or math.isnan(float(value)):
        return "n/a"
    return f"{value:.{digits}f}{unit}"


def markdown_table(pid, mekf, rl, physics):
    rows = [
        ("PID trajectory", "RMS position error", fmt_float(pid.rms_position_error_m, " m")),
        ("PID trajectory", "Mean position error", fmt_float(pid.mean_position_error_m, " m")),
        ("PID trajectory", "Max position error", fmt_float(pid.max_position_error_m, " m")),
        ("PID trajectory", "Final landing error", fmt_float(pid.final_position_error_m, " m")),
        ("MEKF attitude", "Attitude RMSE", fmt_float(mekf.attitude_rmse_deg, " deg", 2)),
        ("MEKF attitude", "95th percentile attitude error", fmt_float(mekf.attitude_p95_deg, " deg", 2)),
        ("MEKF attitude", "Roll / pitch / yaw RMSE", (
            f"{fmt_float(mekf.roll_rmse_deg, '', 2)} / "
            f"{fmt_float(mekf.pitch_rmse_deg, '', 2)} / "
            f"{fmt_float(mekf.yaw_rmse_deg, '', 2)} deg"
        )),
        ("RL recovery", "Success rate", f"{rl.successes}/{rl.episodes} ({rl.success_rate_pct:.1f}%)"),
        ("RL recovery", "Crash rate", f"{rl.crashes}/{rl.episodes} ({rl.crash_rate_pct:.1f}%)"),
        ("RL recovery", "Mean recovery time", fmt_float(rl.mean_recovery_time_s, " s", 2)),
        ("RL recovery", "Max recovered initial tilt", fmt_float(rl.max_recovered_initial_tilt_deg, " deg", 1)),
        ("Physics core", "C++ RK4 throughput", f"{physics.steps_per_second:,.0f} steps/s"),
    ]
    lines = [
        "| Area | Metric | Result |",
        "|------|--------|--------|",
    ]
    lines += [f"| {area} | {metric} | {result} |" for area, metric, result in rows]
    return "\n".join(lines)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def main():
    default_policy = Path(__file__).with_name("quad_recovery_policy")
    parser = argparse.ArgumentParser(description="Run reproducible quad-sim benchmarks.")
    parser.add_argument("--duration", type=float, default=28.0, help="PID/MEKF trajectory duration in seconds")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rl-episodes", type=int, default=200)
    parser.add_argument("--policy", default=str(default_policy), help="Path to PPO policy, with or without .zip")
    parser.add_argument("--stochastic", action="store_true", help="Evaluate stochastic RL policy actions")
    parser.add_argument("--speed-steps", type=int, default=100000)
    parser.add_argument("--markdown", help="Optional path to write the Markdown table")
    parser.add_argument("--json", help="Optional path to write machine-readable results")
    args = parser.parse_args()

    pid, mekf = run_pid_mekf(args.duration, args.seed)
    rl = run_rl(args.rl_episodes, args.seed, args.policy, args.stochastic)
    physics = run_physics_speed(args.speed_steps)

    table = markdown_table(pid, mekf, rl, physics)
    print("\nquad-sim benchmark")
    print("==================")
    print(f"seed: {args.seed}")
    print(f"PID/MEKF duration: {args.duration:.1f}s")
    print(f"RL episodes: {args.rl_episodes}")
    print()
    print(table)

    if args.markdown:
        write_text(args.markdown, table)
        print(f"\nwrote {args.markdown}")

    if args.json:
        payload = {
            "seed": args.seed,
            "pid": asdict(pid),
            "mekf": asdict(mekf),
            "rl": asdict(rl),
            "physics": asdict(physics),
        }
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
