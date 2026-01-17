import gymnasium as gym
from sts_env import StsEnv
import numpy as np

def main():
    print("Checking StsEnv Integration...")
    
    # 1. Instantiate
    env = StsEnv()
    print("Environment Created.")
    
    # 2. Reset
    obs, info = env.reset(seed=42)
    print(f"Reset Complete. Obs Shape: {obs.shape}")
    print(f"Initial HP: {env.gc.cur_hp}/{env.gc.max_hp}")
    
    # 3. Step Loop (Random)
    print("Starting Random Step Loop...")
    total_reward = 0
    
    for i in range(50):
        # Pick random action
        action = env.action_space.sample()
        
        # Override for testing: 
        # If in Battle (state 9), try to play cards (0-9) or End Turn (10)
        # If not, use 11 (Proceed)
        state_val = int(env.gc.screen_state) if not hasattr(env.gc.screen_state, 'value') else env.gc.screen_state.value
        
        if state_val == 9: # Battle
            if np.random.rand() < 0.2:
                action = 10 # End Turn
            else:
                action = np.random.randint(0, 10) # Play Card
        else:
            action = 11 # Proceed
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        step_type = "Valid"
        if "error" in info:
            step_type = "INVALID"
            
        print(f"Step {i+1}: Action {action} -> {step_type} | Reward: {reward:.2f} | Info: {info}")
        
        if terminated or truncated:
            print("Episode Finished.")
            break
            
    print(f"Check Complete. Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
