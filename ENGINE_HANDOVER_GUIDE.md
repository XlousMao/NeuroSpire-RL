# ENGINE_HANDOVER_GUIDE.md

> **Status**: APPROVED for 60M Step PPO Training  
> **Architect**: Trae (Chief System Architect Mode)  
> **Date**: 2026-01-17  

---

## 1. Project Overview (Project Map)

This project is a high-performance Reinforcement Learning environment for *Slay the Spire*, built on a C++ core with Python bindings.

### 1.1 Directory Structure & Responsibilities

```mermaid
graph TD
    Root[sts_lightspeed]
    Root --> Src[src/ (C++ Core Engine)]
    Root --> Bindings[bindings/ (Python Interface)]
    Root --> Include[include/ (Headers)]
    Root --> Scripts[Python Training Scripts]

    Src --> Game[game/ - GameContext, Map, RNG]
    Src --> Combat[combat/ - BattleContext, Cards, Monsters]
    
    Bindings --> PyBind[slaythespire.cpp - PyBind11 Glue]
    
    Scripts --> Env[sts_env.py - Gym Wrapper & Guardrails]
    Scripts --> Train[train_ppo.py - SB3 Training Loop]
    Scripts --> Audit[final_gate_test.py - Logic Verification]
```

### 1.2 Core Class Models

*   **`GameContext` (The God Object)**
    *   **Role**: Manages the entire global state (Floor, Act, HP, Deck, Relics, RNG).
    *   **Key State**: `screenState` (determines valid inputs), `regainControlAction` (callback for returning to neutral state).
    *   **Data Flow**: Python calls methods like `play_card` -> C++ updates `GameContext` -> Python reads `get_observation()` (flat vector).

*   **`Map` & Coordinate System**
    *   **Structure**: 2D Grid (`MapNode`).
    *   **Coordinates**: `curMapNodeX` (0-6), `curMapNodeY` (0-15).
    *   **Verticality**: 
        *   `Y = 0-14`: Standard Rooms.
        *   `Y = 15`: Boss Room.
        *   **CRITICAL**: On Act Transition, `Y` MUST reset to `-1`.

*   **`BattleContext`**
    *   **Role**: Transient state for combat. Created on `enterBattle`, destroyed on `afterBattle`.
    *   **Safety**: Pointers to `BattleContext` in Python are **invalid** after combat ends. Always check `screen_state == BATTLE`.

---

## 2. Deep Diagnosis: The "Floor 115" & Y=15 Crash

### 2.1 The Root Cause
In previous iterations, the AI encountered a **CRITICAL ERROR** or infinite loop at the end of Act 1.
*   **Symptoms**: `curMapNodeY` remained at `15` (Boss Room), but the game state drifted back to `MAP_SCREEN`.
*   **Trigger**: The AI issued a `choose_map_node` command.
*   **Result**: `transitionToMapNode` attempted `++curMapNodeY` (15 -> 16), triggering an out-of-bounds access on the Map vector (Max size 15).
*   **Fix**: 
    1.  **C++ Layer**: In `GameContext::regainControl`, added logic to detect `Y=15`. If found, force transition to `BOSS_RELIC_REWARDS` or Next Act instead of defaulting to `MAP_SCREEN`.
    2.  **Python Layer**: In `sts_env.py`, `_force_unstuck` now explicitly forbids `choose_map_node` if `Y >= 15`.

### 2.2 Suicide Code Audit (High Risk Areas)
The following patterns in `src/` are dangerous for RL training and should be treated with caution:

*   **`assert(false)` / `assert(...)`**: 
    *   *Usage*: Found extensively in `GameContext.cpp` (e.g., Event handling switch cases).
    *   *Risk*: Terminates the entire training process instantly.
    *   *Mitigation*: In Release builds, these are often compiled out, but logical holes remain. **Recommendation**: Replace critical asserts with `return` or throw exceptions that `sts_env.py` can catch to trigger `env.reset()`.

*   **`std::cerr << "CRITICAL ERROR..."`**:
    *   *Location*: `GameContext::transitionToMapNode`.
    *   *Purpose*: Last-line defense against OOB map access.
    *   *Status*: **Active**. If you see this log, the Action Guardrails have failed.

---

## 3. Action Guardrails (The "Safety Net")

To prevent the AI from crashing the C++ engine, `sts_env.py` enforces the following **Illegal Action Masks**:

### 3.1 The "No-Fly Zone" Rules
| Screen State | Forbidden Actions | Reason |
| :--- | :--- | :--- |
| **BATTLE** | `choose_map_node`, `choose_event`, etc. | Context is locked to Combat. |
| **MAP** | `play_card`, `end_turn` | No battle context exists. |
| **BOSS ROOM (Y=15)** | `choose_map_node` | **CRITICAL**. There is no "next node" on the current map. Must transition Act. |
| **REWARDS** | `play_card` | Invalid context. |

### 3.2 Defensive Protocol in `step()`
1.  **Pre-Check**: Before passing action to C++, validate against `screen_state`.
2.  **Try-Catch**: Wrap ALL C++ calls in `try...except`.
3.  **Fail-Safe**: If an exception occurs, **DO NOT CRASH**. Return `reward = 0`, `terminated = True`, and let `env.reset()` clean up.

---

## 4. Connectivity Verification & Roadmap

### 4.1 Verification: `final_gate_test.py`
Before any major training run, execute this script to physically prove:
1.  **Act 2 -> 3**: Teleport to Floor 33, kill boss, verify Floor 34 + `Y = -1`.
2.  **Act 3 -> Victory**: Teleport to Floor 50, kill boss, verify Victory Screen.
3.  **Reset Cleanliness**: Verify `env.reset()` restores HP to Max and Floor to 0.

### 4.2 Robustness Roadmap (Top Priorities)
1.  **Deterministic RNG Seed Logging**: Ensure every crash can be reproduced by logging the exact `seed` used in `env.reset()`.
2.  **Global Exception Decorator**: Wrap the C++ binding functions in a decorator that translates C++ `std::runtime_error` to Python `RuntimeError` for cleaner handling.
3.  **Stuck Detection 2.0**: Implement a "State Hash" history. If the game state (HP + Floor + Screen) hasn't changed in 50 steps, force `regain_control` or kill the episode.

---

## 5. Architect's Note: How to Clear the Spire

To the developer taking over:

The **Strength** of this engine is its raw speed (3000+ FPS). It cheats time by running the game logic directly in C++ memory.

The **Weakness** is its fragility. The original Slay the Spire codebase relies heavily on "Managers" and visual state transitions. We have stripped the visuals, but the *logical glue* (callbacks, `regainControl`) is brittle. One missed callback, and the state machine hangs.

**To Win:**
Don't fight the fragility—**manage it**. 
*   Trust the **PPO** algorithm to find the strategy.
*   Trust **`sts_env.py`** to catch the crashes.
*   Your job is to ensure that when the engine *does* trip (and it will), it stands back up (Reset) instantly, without stopping the training loop.

*The Spire sleeps. Wake it up.*

**- Trae**
