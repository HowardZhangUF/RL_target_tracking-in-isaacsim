from isaacsim import SimulationApp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium as gym
import numpy as np
import os


from ppo_env import LeaderFollowerEnv


gym.register(
    id='LeaderFollower-v0',
    entry_point='ppo_env:LeaderFollowerEnv',
    max_episode_steps=1000,
)


env = gym.make('LeaderFollower-v0')
env.red_area_rect = (1.0, 2.0, -1.0, 2.0)
env.occlusion_mode = "leader_inside"


model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="C:/isaacsim/standalone_examples/custom_env/leader_follower_logs",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
)


print("Starting training...")
model.learn(total_timesteps=100_000)


model_save_path = "C:\isaacsim\standalone_examples\custom_env\leader_follower_ppo_model_pf_nooccl"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")


env.close()