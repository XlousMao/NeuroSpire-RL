import numpy as np

# Fixed dimension for RL observation
OBSERVATION_SIZE = 72 

def get_observation(gc):
    """
    Converts GameContext into a fixed-dimension feature vector for Reinforcement Learning.
    
    Layout (72 floats):
    [0-8]:   Player State (HP%, Block, Energy, Str, Dex, Vuln, Weak, Frail, Artifact)
    [9-33]:  Monsters (5 slots * 5 features: HP%, IntentID, Dmg, Hits, Alive)
    [34-63]: Hand (10 slots * 3 features: ID, Cost, Upgraded)
    [64-71]: Global (Draw/50, Disc/50, Exh/50, Floor/50, Gold/1k, Corruption, DarkEmb, DeadBranch)
    
    Returns:
        np.array of shape (72,), dtype=np.float32
    """
    
    # 1. Get Raw Data from C++ Binding (High Performance)
    try:
        data = gc.get_observation_props()
    except Exception as e:
        # Fallback for safety
        print(f"Error getting observation props: {e}")
        return np.zeros(OBSERVATION_SIZE, dtype=np.float32)

    vec = []
    
    # --- Player State (9 features) ---
    # HP Ratio
    max_hp = data.get("max_hp", 1)
    if max_hp <= 0: max_hp = 1
    vec.append(data.get("cur_hp", 0) / max_hp)
    
    # Raw Stats
    vec.append(float(data.get("block", 0)))
    vec.append(float(data.get("energy", 0)))
    
    # Buffs
    vec.append(float(data.get("strength", 0)))
    vec.append(float(data.get("dexterity", 0)))
    vec.append(float(data.get("vulnerable", 0)))
    vec.append(float(data.get("weak", 0)))
    vec.append(float(data.get("frail", 0)))
    vec.append(float(data.get("artifact", 0)))
    
    # --- Monster State (5 slots * 5 features = 25) ---
    monsters = data.get("monsters", [])
    for i in range(5):
        if i < len(monsters):
            m = monsters[i]
            # HP%
            m_max = m.get("max_hp", 1)
            if m_max <= 0: m_max = 1
            vec.append(m.get("cur_hp", 0) / m_max)
            
            # Intent (Integer ID)
            vec.append(float(m.get("intent_id", 0)))
            
            # Intent Damage (Raw)
            vec.append(float(m.get("intent_dmg", 0)))
            
            # Intent Hits (Raw)
            vec.append(float(m.get("intent_hits", 0)))
            
            # Is Alive (Boolean)
            vec.append(1.0 if m.get("is_alive", False) else 0.0)
        else:
            # Padding
            vec.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            
    # --- Hand Cards (10 slots * 3 features = 30) ---
    hand = data.get("hand", [])
    for i in range(10):
        if i < len(hand):
            c = hand[i]
            # Card ID (Integer Enum)
            vec.append(float(c.get("id", 0)))
            
            # Cost (Raw)
            vec.append(float(c.get("cost", 0)))
            
            # Upgraded (Boolean)
            vec.append(float(c.get("upgraded", 0)))
        else:
            # Padding
            vec.extend([0.0, 0.0, 0.0])
            
    # --- Global Features (8 features) ---
    # Piles (Normalized by approx max size 50)
    vec.append(data.get("draw_pile_size", 0) / 50.0)
    vec.append(data.get("discard_pile_size", 0) / 50.0)
    vec.append(data.get("exhaust_pile_size", 0) / 50.0)
    
    # Floor (Normalized by approx max floor 50)
    vec.append(data.get("floor_num", 0) / 50.0)
    
    # Gold (Normalized by approx max gold 1000)
    vec.append(data.get("gold", 0) / 1000.0)
    
    # Core Relics/Powers (Booleans)
    vec.append(1.0 if data.get("has_corruption", False) else 0.0)
    vec.append(1.0 if data.get("has_dark_embrace", False) else 0.0)
    vec.append(1.0 if data.get("has_dead_branch", False) else 0.0)
    
    return np.array(vec, dtype=np.float32)
