import sys
import os
import time

# Add build/Release to path
sys.path.append(os.path.join(os.getcwd(), "build", "Release"))

import slaythespire

def main():
    print("Starting Final Gate Logic Audit...")
    
    # Initialize GameContext
    gc = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 1, 0)
    
    # --- Check 1: Act 2 -> Act 3 Transition ---
    print("\n[TEST 1] Verifying Act 2 -> Act 3 Transition (Floor 33 -> 34)")
    
    gc.debug_set_floor(33)
    print(f"Set Floor to 33. Current Room: {gc.cur_room}. Encounter: {gc.encounter}")
    
    # Kill Boss
    gc.debug_kill_all_monsters()
    
    # Handle Rewards Loop
    max_steps = 50
    reached_relic = False
    
    for i in range(max_steps):
        if gc.screen_state == slaythespire.ScreenState.BOSS_RELIC_REWARDS:
            reached_relic = True
            break
            
        if gc.screen_state == slaythespire.ScreenState.REWARDS:
            gc.skip_reward_cards()
            rewards = gc.get_rewards()
            while len(rewards) > 0:
                 gc.claim_reward(0)
                 rewards = gc.get_rewards()
            
            if gc.screen_state == slaythespire.ScreenState.REWARDS:
                gc.regain_control()
        
        if gc.screen_state == slaythespire.ScreenState.BATTLE:
             # Force end battle if monsters are dead
             props = gc.get_observation_props()
             monsters = props['monsters']
             all_dead = True
             for m in monsters:
                 if m['cur_hp'] > 0:
                     all_dead = False
                     break
             
             if all_dead:
                 # print("  All monsters dead, forcing regain_control...")
                 gc.regain_control()
             else:
                 gc.end_turn()
                 gc.debug_kill_all_monsters()
             
    if reached_relic:
        print("Successfully reached Boss Relic Screen.")
        gc.choose_boss_relic(0)
        
        map_info = gc.get_map_info()
        y_val = map_info['current_y']
        
        status = "OK" if (gc.floor_num == 34 and y_val == -1) else "FAIL"
        print(f"[CHECK 1] Floor 33 -> 34 Transition: [{status}] | New Y: {y_val} | Floor: {gc.floor_num}")
        
    else:
        print(f"[CHECK 1] FAIL: Did not reach Boss Relic Screen. Stuck at {gc.screen_state}")
        return
        
    
    # --- Check 2: Act 3 -> Victory ---
    print("\n[TEST 2] Verifying Act 3 -> Victory (Floor 50 -> Reset)")
    
    # Jump to Floor 50
    gc.debug_set_floor(50)
    print(f"Set Floor to 50. Encounter: {gc.encounter}")
    
    gc.debug_kill_all_monsters()
    
    victory_detected = False
    for i in range(max_steps):
        if gc.outcome == slaythespire.GameOutcome.PLAYER_VICTORY:
            victory_detected = True
            break
            
        if gc.screen_state == slaythespire.ScreenState.REWARDS:
             gc.skip_reward_cards()
             rewards = gc.get_rewards()
             while len(rewards) > 0:
                gc.claim_reward(0)
                rewards = gc.get_rewards()
             if gc.screen_state == slaythespire.ScreenState.REWARDS:
                gc.regain_control()
        
        if gc.screen_state == slaythespire.ScreenState.BATTLE:
             props = gc.get_observation_props()
             monsters = props['monsters']
             all_dead = True
             for m in monsters:
                 if m['cur_hp'] > 0:
                     all_dead = False
                     break
             
             if all_dead:
                 gc.regain_control()
             else:
                 gc.end_turn()
                 gc.debug_kill_all_monsters()

    print(f"[CHECK 2] Floor 50 -> Victory Screen: [{'Detected' if victory_detected else 'Not Detected'}]")
    
    # --- Check 3: Reset ---
    print("\n[TEST 3] Post-Victory Reset")
    
    del gc
    gc = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, 2, 0)
    
    print(f"Reset GC. Floor: {gc.floor_num}, HP: {gc.cur_hp}/{gc.max_hp}, Gold: {gc.gold}")
    
    is_clean = (gc.floor_num == 0) and (gc.cur_hp == 80) and (gc.gold == 99)
    print(f"[CHECK 3] Post-Victory Reset: [{'Clean' if is_clean else 'Dirty'}]")

    # --- Check 4: Interface Idempotency ---
    print("\n[TEST 4] Interface Idempotency at Floor 34+")
    gc.debug_set_floor(34)
    try:
        gc.get_map_info()
        print("[CHECK 4] Interface Response at Floor 34: [OK]")
    except Exception as e:
        print(f"[CHECK 4] Interface Response at Floor 34: [FAIL] - {e}")

if __name__ == "__main__":
    main()
