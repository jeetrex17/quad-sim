#!/usr/bin/env python3
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from quad_env import QuadRecoveryEnv

POLICY_PATH = Path(__file__).with_name("quad_recovery_policy")


def main():
    parser = argparse.ArgumentParser(description="Train PPO attitude recovery policy.")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-out", default=str(POLICY_PATH))
    parser.add_argument("--eval-freq", type=int, default=50_000)
    args = parser.parse_args()

    policy_path = Path(args.policy_out)
    checkpoint_dir = policy_path.parent / "rl_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(QuadRecoveryEnv, n_envs=args.n_envs, seed=args.seed)
    eval_env = Monitor(QuadRecoveryEnv(), info_keywords=("success", "crashed", "tilt_deg"))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir),
        log_path=str(checkpoint_dir),
        eval_freq=max(1, args.eval_freq // args.n_envs),
        n_eval_episodes=50,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs={"net_arch": [128, 128]},
        device="cpu",
    )
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(str(policy_path))
    print(f"saved {policy_path}.zip")
    print(f"best checkpoints in {checkpoint_dir}")


if __name__ == "__main__":
    main()
