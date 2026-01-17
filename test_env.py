
import sys
import os
import time
import gymnasium as gym
from sts_env import StsEnv
import numpy as np

def stress_test():
    print("Starting Stress Test for 'regain control lambda null' fix...")
    
    # Initialize environment directly (single worker)
    env = StsEnv()
    
    # Reset
    obs, info = env.reset(seed=42)
    print("Environment Reset Successful.")
    
    start_time = time.time()
    steps = 0
    max_steps = 10000  # Run for 10k steps to ensure stability
    
    try:
        while steps < max_steps:
            # Simple random agent or semi-intelligent policy
            # We just need to drive the state machine
            
            # Use mask to select valid action
            mask = env.action_masks()
            valid_actions = np.where(mask)[0]
            
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:
                # If no valid actions according to mask, try end turn or proceed
                # This shouldn't happen with proper masking
                action = 10 if env.gc.screen_state == env.SCREEN_BATTLE else 11
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            steps += 1
            if steps % 1000 == 0:
                elapsed = time.time() - start_time
                fps = steps / elapsed
                print(f"Step {steps}/{max_steps} | FPS: {fps:.2f} | Floor: {info.get('floor', '?')} | HP: {info.get('hp_percent', 0)*100:.1f}%")
            
            if terminated or truncated:
                obs, info = env.reset()
                
    except Exception as e:
        print(f"CRITICAL FAILURE: Stress test crashed at step {steps}")
        print(e)
        sys.exit(1)
        
    print("\nStress Test Completed Successfully!")
    print(f"Total Steps: {steps}")
    print(f"Total Time: {time.time() - start_time:.2f}s")
    print("Please verify console logs to ensure no 'regain control lambda was null' errors appeared.")

if __name__ == "__main__":
    stress_test()
