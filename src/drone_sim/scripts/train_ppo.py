#!/usr/bin/env python3
from stable_baselines3 import PPO
from quad_env import QuadRecoveryEnv

env   = QuadRecoveryEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
model.save("quad_recovery_policy")
print("saved quad_recovery_policy.zip")
