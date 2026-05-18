#!/usr/bin/env python3
from pathlib import Path

from stable_baselines3 import PPO
from quad_env import QuadRecoveryEnv

POLICY_PATH = Path(__file__).with_name("quad_recovery_policy")

env = QuadRecoveryEnv()
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=4096,        # larger rollout buffer = more stable gradients
    batch_size=128,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    ent_coef=0.005,      # small entropy bonus keeps exploration alive longer
)
model.learn(total_timesteps=2_000_000)
model.save(str(POLICY_PATH))
print(f"saved {POLICY_PATH}.zip")
