import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Import our custom modules
import sys
import os

# Add build/Release to path to find the compiled extension
pyd_path = os.path.join(os.path.dirname(__file__), "build", "Release")
if os.path.exists(pyd_path):
    sys.path.append(pyd_path)

import slaythespire
import observation
import reward
import map_evaluator

class StsEnv(gym.Env):
    """
    Slay the Spire Gymnasium Environment (Ironclad)
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # Action Space: 15 Discrete Actions
        # 0-9: Play Card in Hand Slot 0-9
        # 10: End Turn
        # 11: Proceed (General / Map / Rewards / Event)
        # 12-14: Reserved (e.g. specialized choices, but for now we map 'Proceed' to auto-handle)
        # Wait, user specified:
        # 11-14: Handle non-combat screens (Map, Rewards, Neow)
        # Let's map 11 to "Auto-Handle Non-Combat" for simplicity in early training.
        # But to follow user instruction strictly: "11-14: 处理非战斗界面"
        # I will treat 11 as the generic "Interact/Proceed" action for non-combat.
        self.action_space = spaces.Discrete(15)
        
        # Observation Space: 72-dim vector
        self.observation_space = spaces.Box(
            low=-5.0, high=500.0, shape=(72,), dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.gc = None
        self.obs_prev = None
        self.step_count = 0
        self.max_steps = 2000
        
        # Screen Constants
        self.SCREEN_INVALID = 0
        self.SCREEN_EVENT = 1
        self.SCREEN_REWARDS = 2
        self.SCREEN_BOSS_RELIC = 3
        self.SCREEN_CARD_SELECT = 4
        self.SCREEN_MAP = 5
        self.SCREEN_TREASURE = 6
        self.SCREEN_REST = 7
        self.SCREEN_SHOP = 8
        self.SCREEN_BATTLE = 9

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize GameContext
        game_seed = seed if seed is not None else random.randint(0, 10000)
        self.gc = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, game_seed, 0)
        
        # Debug Reset
        # print(f"[StsEnv] Reset. Seed: {game_seed}. HP: {self.gc.cur_hp}/{self.gc.max_hp}. Floor: {self.gc.floor_num}. State: {self.gc.screen_state}")
        
        self.step_count = 0
        self.obs_prev = observation.get_observation(self.gc)
        self.last_state_val = -1
        self.stuck_counter = 0
        self.floor_steps = 0 # Track steps per floor to prevent infinite loops
        self.current_floor = 0 # Track current floor locally
        self.last_floor = 0
        self.last_gold = self.gc.gold
        
        return self.obs_prev, {}

    def step(self, action):
        self.step_count += 1
        self.floor_steps += 1 # Increment floor steps
        truncated = False
        terminated = False
        step_reward = 0.0
        info = {}
        
        # 1. Get Current State
        state_obj = self.gc.screen_state
        state = int(state_obj.value) if hasattr(state_obj, "value") else int(state_obj)
        
        # Floor Change Detection
        if self.gc.floor_num != self.current_floor:
             self.current_floor = self.gc.floor_num
             self.floor_steps = 0 # Reset counter on new floor
        
        # --- Reward Calculation (Pre-Action) ---
        # 1. Floor Progression Reward
        current_floor = self.gc.floor_num
        if current_floor > self.last_floor:
            # Base Floor Reward
            step_reward += 1.0
            
            # Boss Room Arrival (Floor 16)
            if current_floor == 16:
                step_reward += 20.0
                
            # Boss Killer (Act 2 Entry: 16 -> 17)
            if self.last_floor == 16 and current_floor == 17:
                step_reward += 100.0
                
        # 2. Checkpoint Bonus (Elite/Mini-boss)
        # Heuristic: Significant gold gain (>30) usually implies Elite/Boss kill
        current_gold = self.gc.gold
        gold_gain = current_gold - self.last_gold
        if gold_gain >= 30:
            # Avoid double counting Boss rewards (which have huge gold)
            # Boss usually gives ~100g. Elite ~30-40g.
            # We give +5.0 for "Elite/Mini-boss".
            # If floor is 16 or 17, we already gave big rewards above, maybe skip?
            # User said: "Non-16 floor but large gold or rare relic"
            if current_floor != 16 and current_floor != 17:
                step_reward += 5.0
                
        # Update Trackers
        self.last_floor = current_floor
        self.last_gold = current_gold

        
        # Stuck Detection
        if state == self.last_state_val:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_state_val = state
            
        # Floor Timeout Check (Prevent stuck in map/rewards for too long)
        # If we stay on same floor for > 300 steps, kill it.
        # But battle might take long? 300 steps is plenty for a battle (usually < 50 cards).
        # Let's be generous: 500 steps.
        if self.floor_steps > 500:
             truncated = True
             step_reward -= 5.0 # Penalty for wasting time
             info["error"] = "Floor Timeout"
             
        # 2. Handle Action based on State
        action_valid = False
        
        if state == self.SCREEN_BATTLE:
            # Battle Logic
            if 0 <= action <= 9:
                # Play Card
                hand = self.gc.hand
                if action < len(hand):
                    card = hand[action]
                    cost = card.get('costForTurn', 99) 
                    # Energy is at index 2 in observation vector
                    current_energy = self.obs_prev[2]
                    
                    if cost <= current_energy:
                        alive_monsters = self._get_alive_monsters()
                        target_idx = 0
                        if alive_monsters:
                            target_idx = random.choice(alive_monsters)
                        
                        try:
                            self.gc.play_card(int(action), target_idx)
                            action_valid = True
                        except Exception as e:
                            action_valid = False
                    else:
                        action_valid = False
                else:
                    action_valid = False
                    
            elif action == 10:
                self.gc.end_turn()
                action_valid = True
                
            else:
                action_valid = False
                
        else:
            # Non-Combat Logic
            if action >= 11:
                # If we are stuck, force random choices or specific recovery
                if self.stuck_counter > 5:
                    # Force a random interactions to break loops
                    # print(f"[StsEnv] Stuck in state {state} for {self.stuck_counter} steps. Forcing random action.")
                    self._force_unstuck(state)
                    action_valid = True # We handled it
                else:
                    self._handle_non_combat(state)
                    action_valid = True
            else:
                action_valid = False
        
        # 3. Invalid Action Penalty
        if not action_valid:
            step_reward -= 1.0 # Increased penalty as requested
            info["error"] = "Invalid Action"
        else:
            # Valid Action Reward/Cost
            # Introduce Time Cost: -0.002 per step (Reduced from -0.01)
            step_reward -= 0.002 
            
            # 4. Calculate Rational Reward
            obs_curr = observation.get_observation(self.gc)
            r_rational, r_info = reward.calculate_rational_reward(self.gc, self.obs_prev, obs_curr)
            step_reward += r_rational
            info.update(r_info)
            self.obs_prev = obs_curr

        # 5. Check Termination
        if self.gc.outcome == slaythespire.GameOutcome.PLAYER_VICTORY:
            terminated = True
            step_reward += 500.0
            info["victory"] = True

        if self.gc.cur_hp <= 0:
            terminated = True
            step_reward -= 10.0
            
        if self.step_count >= self.max_steps:
            truncated = True
            
        # Add Floor and HP to Info for Monitoring
        info["floor"] = self.gc.floor_num
        info["hp_percent"] = self.gc.cur_hp / self.gc.max_hp if self.gc.max_hp > 0 else 0
        
        # --- Heartbeat & Debug Logging ---
        # Print critical stats every 100 steps or if special condition
        if self.step_count % 1000 == 0: # Reduce frequency to avoid log spam
             print(f"[StsEnv-{id(self)}] Step: {self.step_count} | Floor: {self.gc.floor_num} | HP: {self.gc.cur_hp}/{self.gc.max_hp}")
        
        # Verify HP consistency
        # if self.gc.cur_hp == self.gc.max_hp and self.gc.floor_num > 5:
             # print(f"[Debug] Perfect HP at Floor {self.gc.floor_num}. HP: {self.gc.cur_hp}")
             
        return self.obs_prev, step_reward, terminated, truncated, info

    def action_masks(self):
        """
        Returns a boolean mask of valid actions for the current state.
        True = Valid, False = Invalid.
        Used by MaskablePPO.
        """
        mask = np.zeros(15, dtype=bool)
        
        state_obj = self.gc.screen_state
        state = int(state_obj.value) if hasattr(state_obj, "value") else int(state_obj)
        
        if state == self.SCREEN_BATTLE:
            # Battle Actions (0-10)
            
            # 1. Card Actions (0-9)
            hand = self.gc.hand
            # Energy is at index 2 in observation vector
            current_energy = self.obs_prev[2]
            
            for i in range(10):
                if i < len(hand):
                    card = hand[i]
                    cost = card.get('costForTurn', 99)
                    if cost <= current_energy:
                        # Also check if card is playable (is_playable field?)
                        # For now, energy check is the main constraint
                        mask[i] = True
            
            # 2. End Turn (10)
            # Always valid in battle to end turn
            mask[10] = True
            
        else:
            # Non-Combat Actions (11-14)
            # Currently we map all non-combat interactions to Action 11 (Proceed/Interact)
            # or heuristics.
            # To allow the agent to learn, we enable Action 11 when not in battle.
            mask[11] = True
            
            # If we wanted to differentiate (e.g. 11=Left, 12=Right), we would need 
            # more complex logic here. For now, 11 is the "Do Something Valid" button.
            
        return mask

    def _force_unstuck(self, state):
        # Try brute force options to proceed
        try: self.gc.choose_neow_option(random.randint(0,3)) 
        except: pass
        try: self.gc.choose_event_option(random.randint(0,2))
        except: pass
        try: self.gc.choose_campfire_option(0)
        except: pass
        
        # Safety: Don't choose map node if at Boss Room (Y=15)
        if self.gc.cur_map_node_y < 15:
            try: self.gc.choose_map_node(random.randint(0,6))
            except: pass
            
        try: self.gc.claim_reward(random.randint(0,5))
        except: pass
        try: 
            # Check if lambda is null implicitly by catching error if possible
            # In C++, regainControl checks for null and prints to stderr but doesn't crash if asserted out.
            # However, if it returns void and does nothing, we are stuck.
            # If we suspect null lambda, we might need to reset screen state manually?
            # We can't access regainControlAction directly from python.
            self.gc.regain_control()
        except: pass

    def _get_alive_monsters(self):
        # Helper to find alive monster indices from observation or gc
        # We can parse the observation vector (Indices 9-33)
        # 9 + i*5 + 4 is 'is_alive'
        alive = []
        # obs is numpy array
        obs = self.obs_prev
        for i in range(5):
            is_alive = obs[9 + i*5 + 4]
            if is_alive > 0.5:
                alive.append(i)
        return alive

    def _handle_non_combat(self, state):
        """
        Delegates non-combat logic to heuristics/MapEvaluator
        """
        # print(f"Handling Non-Combat State: {state}")
        
        handled = False
        
        if state == self.SCREEN_MAP:
            evaluator = map_evaluator.MapEvaluator(self.gc)
            best_x, _ = evaluator.evaluate_path()
            try:
                self.gc.choose_map_node(best_x)
                handled = True
            except:
                # Fallback 1: Try Node 0
                try: 
                    self.gc.choose_map_node(0)
                    handled = True
                except: 
                    # Fallback 2: Try all possible nodes (0-6 are standard widths)
                    for i in range(1, 7):
                        try:
                            self.gc.choose_map_node(i)
                            handled = True
                            break
                        except: pass
                
        elif state == self.SCREEN_REWARDS:
            rewards = self.gc.get_rewards()
            if not rewards:
                self.gc.regain_control()
                handled = True
            else:
                # Simple Heuristic: Take Card if available, else first item
                taken = False
                for i, r in enumerate(rewards):
                    if r['type'] == "CARD":
                        # Pick first card
                        cards = r['cards']
                        if cards:
                            self.gc.pick_reward_card(cards[0])
                            taken = True
                            break
                if not taken:
                    # Take first reward
                    try:
                        self.gc.claim_reward(0)
                        handled = True
                    except: pass

        elif state == self.SCREEN_INVALID: # Neow or Invalid
            try: 
                # Try Neow options
                self.gc.choose_neow_option(0)
                handled = True
            except: 
                try:
                    self.gc.regain_control()
                    handled = True
                except: pass
            
        elif state == self.SCREEN_REST:
            # Simple Heuristic: Rest if HP < 50%, else Smith
            try:
                if (self.gc.cur_hp / self.gc.max_hp) < 0.5:
                    self.gc.choose_campfire_option(0) # Rest
                else:
                    self.gc.choose_campfire_option(1) # Smith
                handled = True
            except:
                try:
                    self.gc.choose_campfire_option(0)
                    handled = True
                except: pass
            
        elif state == self.SCREEN_TREASURE:
            try:
                self.gc.choose_treasure_open()
                handled = True
            except: pass
            
        elif state == self.SCREEN_EVENT:
            try:
                self.gc.choose_event_option(0)
                handled = True
            except: pass
            
        elif state == self.SCREEN_BOSS_RELIC:
            try:
                # Choose random boss relic (0-2)
                # Some bosses might offer fewer? Usually 3.
                self.gc.choose_boss_relic(random.randint(0, 2))
                handled = True
            except: pass
            
        elif state == self.SCREEN_CARD_SELECT:
            try:
                self.gc.choose_card_option(0)
                handled = True
            except: pass
            
        # Default fallback for any other screen (Event, Treasure, etc) or if specific handler failed
        if not handled:
            # Try regain control
            try:
                self.gc.regain_control()
            except:
                pass
            
            # Additional safety: If still not handled and state is INVALID or we are stuck,
            # try to re-initialize or jump to map if possible (hacky)
            # We can't easily force state change without C++ access.
            # But we can try to force a map transition if we are truly lost? No.
            pass

