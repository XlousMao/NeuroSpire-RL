import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from sts_env import StsEnv

def main():
    print("=== Slay the Spire PPO Evaluation Script ===")
    
    model_path = "sts_ppo_final.zip"
    
    # Check if model exists
    try:
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
    except:
        print(f"Model {model_path} not found. Please run training first.")
        return

    # Create Single Environment for Evaluation
    env = StsEnv()
    
    num_episodes = 3
    
    for ep in range(num_episodes):
        print(f"\n--- Episode {ep+1} ---")
        obs, info = env.reset(seed=100 + ep)
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (terminated or truncated):
            steps += 1
            
            # Predict Action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Get details
            floor = info.get('floor', 0)
            hp_pct = info.get('hp_percent', 0.0) * 100
            
            # Print status every step or significant event
            # Action name?
            action_name = "Unknown"
            if action < 10: action_name = f"Play Card {action}"
            elif action == 10: action_name = "End Turn"
            elif action >= 11: action_name = "Non-Combat"
            
            print(f"Floor: {floor} | Action: {action} ({action_name}) | HP: {hp_pct:.1f}% | Step Reward: {reward:.2f}")
            
            # Slow down slightly for readability if needed
            # time.sleep(0.05) 
            
        print(f"Episode {ep+1} Finished. Total Reward: {total_reward:.2f}, Final Floor: {env.gc.floor_num}")

if __name__ == "__main__":
    main()
