import numpy as np

def calculate_rational_reward(gc, obs_prev, obs_curr):
    """
    Calculates a 'Rational' reward for Ironclad Agent.
    Values:
      - HP is a resource: Penalize large losses, but ignore small chip damage if it gains tempo.
      - Efficiency: Penalize wasted block and energy.
      - Scaling: Reward strength gain, deck thinning, and gold accumulation.
      
    Args:
        gc: GameContext (current state object, for complex checks)
        obs_prev: np.array (72,) from previous step
        obs_curr: np.array (72,) from current step (after action)
        
    Returns:
        float: Reward value
        dict: Breakdown of reward components for logging
    """
    
    reward = 0.0
    info = {}
    
    # --- 1. HP Management (Resource View) ---
    # Prev HP% (Index 0), Curr HP% (Index 0)
    hp_prev = obs_prev[0]
    hp_curr = obs_curr[0]
    hp_diff = hp_curr - hp_prev
    
    # Base penalty for losing HP
    # 100% HP loss = -10.0 reward (Death penalty is handled by episode termination usually, but here helps gradient)
    if hp_diff < 0:
        # Non-linear penalty: losing 1 HP is fine (-0.05), losing 20 HP is bad (-2.0)
        # However, for simplicity and stability, linear is often better for PPO.
        # Let's use a coefficient. 
        # Max HP is approx 80. 1 HP ~ 1.25%.
        # Reward = diff * 5.0 -> Losing 10% HP = -0.5.
        r_hp = hp_diff * 5.0 
        
        # "Rational" adjustment: If we have Strength gain in same turn, mitigate penalty?
        # Too complex for single step. Keep it simple.
        reward += r_hp
        info['hp_loss'] = r_hp

    # --- 2. Combat Efficiency (Block & Energy) ---
    # Block (Index 1)
    # Check if we over-blocked? 
    # Hard to tell without knowing incoming damage exactly in vector.
    # But vector has Monster Intents!
    # Monsters: Indices 9-33. Each monster 5 slots.
    # [9]: HP, [10]: Intent, [11]: Dmg, [12]: Hits, [13]: Alive
    
    incoming_dmg = 0
    for i in range(5):
        alive = obs_prev[9 + i*5 + 4]
        if alive > 0.5:
            dmg = obs_prev[9 + i*5 + 2]
            hits = obs_prev[9 + i*5 + 3]
            incoming_dmg += dmg * hits
            
    current_block = obs_curr[1] # Block after action
    
    # If turn ended (we can't easily tell if turn ended just by obs difference, 
    # but let's assume this function is called after every step).
    # Actually, Overblock is only bad at END of turn. 
    # Intermediate block is fine.
    # So we might skip this unless we know it's end of turn.
    # Let's skip over-block for now to avoid noise.
    
    # --- 3. Scaling & Offense ---
    # Strength Gain (Index 3)
    str_prev = obs_prev[3]
    str_curr = obs_curr[3]
    if str_curr > str_prev:
        # High value on Strength for Ironclad
        r_str = (str_curr - str_prev) * 0.2
        reward += r_str
        info['strength_gain'] = r_str
        
    # Monster Kills
    # Count alive monsters
    alive_prev = sum([1 for i in range(5) if obs_prev[9 + i*5 + 4] > 0.5])
    alive_curr = sum([1 for i in range(5) if obs_curr[9 + i*5 + 4] > 0.5])
    
    if alive_curr < alive_prev:
        # Killed a monster!
        killed_count = alive_prev - alive_curr
        r_kill = killed_count * 1.0 # Big reward
        reward += r_kill
        info['kill'] = r_kill
        
    # Damage Dealt (Proxy via Monster HP)
    # Sum of Monster HP%
    # This incentivizes damage even if no kill
    m_hp_sum_prev = sum([obs_prev[9 + i*5] for i in range(5)])
    m_hp_sum_curr = sum([obs_curr[9 + i*5] for i in range(5)])
    
    hp_drop = m_hp_sum_prev - m_hp_sum_curr
    if hp_drop > 0:
        # Reward damage. 
        # Total HP pool is maybe 50-300. 
        # 1% total HP damage = 0.01. 
        # Let's scale it up.
        r_dmg = hp_drop * 2.0
        reward += r_dmg
        info['damage'] = r_dmg

    # --- 4. Long Term Growth (Global Features) ---
    # Gold (Index 68: Gold/1000)
    gold_prev = obs_prev[68]
    gold_curr = obs_curr[68]
    if gold_curr > gold_prev:
        # Gained gold
        r_gold = (gold_curr - gold_prev) * 10.0 # 100 gold = 0.1 diff * 10 = 1.0 reward
        reward += r_gold
        info['gold'] = r_gold
        
    # Floor Climb
    # Floor is usually encoded in observation too, but we can access via gc for reliability if obs is normalized
    # Or use obs[67] (Floor/50)
    # Let's rely on obs diff for speed
    floor_prev = obs_prev[67]
    floor_curr = obs_curr[67]
    if floor_curr > floor_prev:
        # Climbed a floor!
        # 1 floor = 1/50 = 0.02.
        # Reward: +1.0 per floor
        r_floor = (floor_curr - floor_prev) * 50.0 * 1.0 
        reward += r_floor
        info['floor_climb'] = r_floor
        
    # Deck Thinning (Index 69: DeckSize/50)
    deck_prev = obs_prev[69]
    deck_curr = obs_curr[69]
    
    # Check if deck size decreased (Card Removal)
    # Ignore small fluctuations if they are temporary, but deck size usually changes permanently via removal/transform.
    # We want to reward REMOVAL.
    if deck_curr < deck_prev:
        # Removed a card!
        # 1 card = 1/50 = 0.02 diff.
        # Reward removal heavily -> +0.5 per card.
        r_thin = (deck_prev - deck_curr) * 25.0 
        reward += r_thin
        info['deck_thin'] = r_thin
        
    return reward, info

