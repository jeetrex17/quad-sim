#!/usr/bin/env python3
import argparse
import csv
import math
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
from stable_baselines3 import PPO

from quad_env import DT, MAX_STEPS, QuadRecoveryEnv


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    success: bool
    crashed: bool
    return_sum: float
    initial_tilt_deg: float
    final_tilt_deg: float
    final_z_error_m: float
    final_omega_norm_rad_s: float
    recovery_time_s: float
    steps: int


def _tilt_deg(obs):
    qw = float(obs[6])
    return 2.0 * math.degrees(math.acos(np.clip(abs(qw), 0.0, 1.0)))


def _omega_norm(obs):
    return float(np.linalg.norm(obs[10:13]))


def _resolve_policy(path):
    path = Path(path)
    candidates = [path]
    if path.suffix != ".zip":
        candidates.append(path.with_suffix(".zip"))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.with_suffix(""))

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Policy file not found. Train one first with scripts/train_ppo.py.\n"
        f"Checked:\n{checked}"
    )


def _is_stable(obs, tilt_deg_max, omega_max, z_error_max):
    return (
        _tilt_deg(obs) <= tilt_deg_max
        and _omega_norm(obs) <= omega_max
        and abs(float(obs[2]) - 1.0) <= z_error_max
    )


def _run_episode(model, env, episode, seed, args):
    obs, _ = env.reset(seed=seed)
    initial_tilt = _tilt_deg(obs)
    hold_steps = max(1, int(round(args.hold_s / DT)))
    stable_steps = 0
    recovery_time = math.nan
    return_sum = 0.0
    crashed = False
    steps = 0

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, reward, terminated, truncated, _ = env.step(action)
        return_sum += float(reward)
        steps = step + 1

        if _is_stable(obs, args.success_tilt_deg, args.success_omega, args.success_z_error):
            stable_steps += 1
            if stable_steps >= hold_steps:
                recovery_time = (step + 1 - hold_steps + 1) * DT
                break
        else:
            stable_steps = 0

        if terminated:
            crashed = True
            break
        if truncated:
            break

    success = not math.isnan(recovery_time)
    return EpisodeResult(
        episode=episode,
        seed=seed,
        success=success,
        crashed=crashed and not success,
        return_sum=return_sum,
        initial_tilt_deg=initial_tilt,
        final_tilt_deg=_tilt_deg(obs),
        final_z_error_m=abs(float(obs[2]) - 1.0),
        final_omega_norm_rad_s=_omega_norm(obs),
        recovery_time_s=recovery_time,
        steps=steps,
    )


def _mean(values):
    values = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(values)) if values else math.nan


def _percent(n, d):
    return 100.0 * n / d if d else 0.0


def _print_summary(results, args):
    total = len(results)
    successes = [r for r in results if r.success]
    crashes = [r for r in results if r.crashed]

    print("\nRL recovery evaluation")
    print("----------------------")
    print(f"episodes:       {total}")
    print(f"success:        {len(successes)}/{total} ({_percent(len(successes), total):.1f}%)")
    print(f"crashes:        {len(crashes)}/{total} ({_percent(len(crashes), total):.1f}%)")
    print(f"criteria:       tilt <= {args.success_tilt_deg:.1f} deg, "
          f"|omega| <= {args.success_omega:.2f} rad/s, "
          f"|z-1m| <= {args.success_z_error:.2f} m for {args.hold_s:.2f}s")

    if successes:
        print(f"mean recovery:  {_mean(r.recovery_time_s for r in successes):.2f} s")
        print(f"max recovered:  {max(r.initial_tilt_deg for r in successes):.1f} deg initial tilt")
    else:
        print("mean recovery:  n/a")
        print("max recovered:  n/a")

    print(f"mean final tilt:{_mean(r.final_tilt_deg for r in results):7.2f} deg")
    print(f"mean z error:   {_mean(r.final_z_error_m for r in results):7.3f} m")
    print(f"mean |omega|:   {_mean(r.final_omega_norm_rad_s for r in results):7.3f} rad/s")

    bins = [(0, 25), (25, 45), (45, 65), (65, 85)]
    print("\nSuccess by initial tilt")
    for lo, hi in bins:
        bucket = [r for r in results if lo <= r.initial_tilt_deg < hi]
        ok = sum(r.success for r in bucket)
        print(f"  {lo:02d}-{hi:02d} deg: {ok:3d}/{len(bucket):3d} ({_percent(ok, len(bucket)):5.1f}%)")


def _write_csv(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    print(f"\nwrote {path}")


def main():
    default_policy = Path(__file__).with_name("quad_recovery_policy")
    parser = argparse.ArgumentParser(description="Evaluate PPO quadrotor recovery policy.")
    parser.add_argument("--policy", default=str(default_policy), help="Path to SB3 PPO policy, with or without .zip")
    parser.add_argument("--episodes", type=int, default=200, help="Number of randomized recovery trials")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions")
    parser.add_argument("--success-tilt-deg", type=float, default=12.0)
    parser.add_argument("--success-omega", type=float, default=0.8)
    parser.add_argument("--success-z-error", type=float, default=0.35)
    parser.add_argument("--hold-s", type=float, default=0.20, help="How long the recovered state must stay stable")
    parser.add_argument("--csv", help="Optional path for per-episode CSV output")
    args = parser.parse_args()

    policy_path = _resolve_policy(args.policy)
    env = QuadRecoveryEnv()
    model = PPO.load(policy_path)

    results = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        results.append(_run_episode(model, env, episode, seed, args))

    print(f"policy:         {policy_path}.zip")
    _print_summary(results, args)
    if args.csv:
        _write_csv(args.csv, results)


if __name__ == "__main__":
    main()
