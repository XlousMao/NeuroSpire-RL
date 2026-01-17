
import time
import sys
import numpy as np
from sts_env import StsEnv

def logic_audit():
    print("Starting Logic Audit (Normal Mode)...")
    
    env = StsEnv()
    obs, info = env.reset(seed=42)
    
    stats = {
        "max_floor": 0,
        "victory_count": 0,
        "act2_transition_count": 0,
        "act3_transition_count": 0,
    }
    
    prev_floor = env.gc.floor_num
    start_time = time.time()
    
    # Run for 100,000 steps to verify stability without insta-kill
    for step in range(1, 100001):
        
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
        else:
            action = 11 # Proceed
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_floor = env.gc.floor_num
        if current_floor > stats["max_floor"]:
            stats["max_floor"] = current_floor
            
        if prev_floor == 16 and current_floor == 17:
            print(f"[SUCCESS] Act 1 -> 2 Transition at Step {step}")
            stats["act2_transition_count"] += 1
            
        if prev_floor == 33 and current_floor == 34:
             print(f"[SUCCESS] Act 2 -> 3 Transition at Step {step}")
             stats["act3_transition_count"] += 1
        
        if info.get("victory"):
             print(f"[VICTORY] Game Completed at Step {step}!")
             stats["victory_count"] += 1
             
        if step % 5000 == 0:
             print(f"Step {step} | Floor {current_floor} | Max Floor {stats['max_floor']}")

        prev_floor = current_floor
        
        if terminated or truncated:
            obs, info = env.reset()
            prev_floor = env.gc.floor_num
            
    end_time = time.time()
    print("\n" + "="*40)
    print("LOGIC AUDIT RESULTS")
    print(f"Total Steps: 100,000")
    print(f"Max Floor: {stats['max_floor']}")
    print(f"Victories: {stats['victory_count']}")
    print("="*40)

if __name__ == "__main__":
    logic_audit()
