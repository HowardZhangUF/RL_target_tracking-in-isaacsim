from isaacsim import SimulationApp
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import time


from ppo_env import LeaderFollowerEnv

# Register the environment with Gym
gym.register(
    id='LeaderFollower-v0',
    entry_point='ppo_env:LeaderFollowerEnv',
    max_episode_steps=600,
)

def test_ppo_model():
    """Test the environment using the trained PPO model"""
    env = gym.make('LeaderFollower-v0')
    
    
    model = PPO.load("C:\isaacsim\standalone_examples\custom_env\leader_follower_ppo_model_v1")
    
    obs, info = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        
        distance = np.linalg.norm(obs)
        print(f"Distance: {distance:.2f}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
        
        if terminated or truncated:
            done = True
            print(f"Episode finished with total reward: {total_reward:.2f}")
            print(f"Final distance: {distance:.2f}")
    
    env.close()

if __name__ == "__main__":
    test_ppo_model()