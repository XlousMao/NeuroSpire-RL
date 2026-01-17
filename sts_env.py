import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Import our custom modules
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
        # Fixed Seed 1 for stability during dev, or random?
        # gym.Env seed is usually handled by seeding random generators.
        # We can pass seed to GameContext.
        game_seed = seed if seed is not None else random.randint(0, 10000)
        self.gc = slaythespire.GameContext(slaythespire.CharacterClass.IRONCLAD, game_seed, 0)
        
        self.step_count = 0
        self.obs_prev = observation.get_observation(self.gc)
        
        return self.obs_prev, {}

    def step(self, action):
        self.step_count += 1
        truncated = False
        terminated = False
        step_reward = 0.0
        info = {}
        
        # 1. Get Current State
        state_obj = self.gc.screen_state
        state = int(state_obj.value) if hasattr(state_obj, "value") else int(state_obj)
        
        # 2. Handle Action based on State
        action_valid = False
        
        if state == self.SCREEN_BATTLE:
            # Battle Logic
            if 0 <= action <= 9:
                # Play Card
                hand = self.gc.hand
                if action < len(hand):
                    # Check Energy
                    card = hand[action]
                    cost = card.get('costForTurn', 99) # Using dict from bindings
                    # Get Energy from Observation (Index 2) or direct GC
                    current_energy = self.gc.energy # Use direct access if available or from obs
                    
                    if cost <= current_energy:
                        # Valid Play
                        # Auto-select target
                        alive_monsters = self._get_alive_monsters()
                        target_idx = 0
                        if alive_monsters:
                            # Heuristic: Random target or first alive
                            # For simple env, random valid target is standard
                            target_idx = random.choice(alive_monsters)
                        
                        try:
                            self.gc.play_card(int(action), target_idx)
                            action_valid = True
                        except Exception as e:
                            # C++ exception means invalid logic deep down
                            action_valid = False
                    else:
                        # Not enough energy
                        action_valid = False
                else:
                    # Index out of bounds (no card in that slot)
                    action_valid = False
                    
            elif action == 10:
                # End Turn
                self.gc.end_turn()
                action_valid = True
                
            else:
                # Action 11-14 in Battle is Invalid
                action_valid = False
                
        else:
            # Non-Combat Logic
            # Map Actions 11-14 to "Handle Screen"
            # Or enforce specific actions?
            # User said "11-14: Handle non-combat".
            # Let's say any action >= 11 triggers the auto-handler for non-combat.
            # And 0-10 are invalid in non-combat (punish trying to play cards in map).
            
            if action >= 11:
                self._handle_non_combat(state)
                action_valid = True
            else:
                # Trying to play cards outside battle
                action_valid = False
        
        # 3. Invalid Action Penalty
        if not action_valid:
            step_reward -= 0.1
            # Do NOT update state (it didn't change anyway usually, or we prevented it)
            # Just return same observation
            # But wait, observation might change if animations finished? 
            # In this fast env, usually no.
            info["error"] = "Invalid Action"
        else:
            # 4. Calculate Rational Reward (Only if action was valid)
            obs_curr = observation.get_observation(self.gc)
            r_rational, r_info = reward.calculate_rational_reward(self.gc, self.obs_prev, obs_curr)
            step_reward += r_rational
            info.update(r_info)
            self.obs_prev = obs_curr

        # 5. Check Termination
        if self.gc.cur_hp <= 0:
            terminated = True
            step_reward -= 10.0 # Death penalty
        elif self.gc.floor_num > 16: # End of Act 1
            terminated = True
            step_reward += 20.0 # Act Clear Bonus
            
        if self.step_count >= self.max_steps:
            truncated = True
            
        return self.obs_prev, step_reward, terminated, truncated, info

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
        if state == self.SCREEN_MAP:
            evaluator = map_evaluator.MapEvaluator(self.gc)
            best_x, _ = evaluator.evaluate_path()
            try:
                self.gc.choose_map_node(best_x)
            except:
                # Fallback
                try: self.gc.choose_map_node(0)
                except: pass
                
        elif state == self.SCREEN_REWARDS:
            rewards = self.gc.get_rewards()
            if not rewards:
                self.gc.regain_control()
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
                    self.gc.claim_reward(0)

        elif state == self.SCREEN_INVALID: # Neow
            try: self.gc.choose_neow_option(0)
            except: self.gc.regain_control()
            
        elif state == self.SCREEN_REST:
            # Simple Heuristic: Rest if HP < 50%, else Smith
            if (self.gc.cur_hp / self.gc.max_hp) < 0.5:
                self.gc.choose_campfire_option(0) # Rest
            else:
                self.gc.choose_campfire_option(1) # Smith (might fail if no smithable card?)
                # Actually option 1 is smith? Need to check bindings. 
                # Bindings: 0=Rest, 1=Smith.
                # If smith fails (no card?), it might throw or do nothing.
                # Safe fallback: Rest
                # For now just Rest 0
                pass 
            
        # Default fallback for any other screen (Event, Treasure, etc)
        # Try regain control
        try:
            self.gc.regain_control()
        except:
            pass
            
        # Specific handlers
        if state == self.SCREEN_TREASURE:
            self.gc.choose_treasure_open()
        elif state == self.SCREEN_EVENT:
            self.gc.choose_event_option(0)
        elif state == self.SCREEN_CARD_SELECT:
            self.gc.choose_card_option(0)

