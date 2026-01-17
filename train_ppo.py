import os
import time
import torch
import numpy as np
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback

# Import our custom environment
import sys
# Ensure build/Release is in path
pyd_path = os.path.join(os.path.dirname(__file__), "build", "Release")
if os.path.exists(pyd_path) and pyd_path not in sys.path:
    sys.path.append(pyd_path)
    
from sts_env import StsEnv

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log floor number and HP from info dict when episode ends
        for info in self.locals['infos']:
            if "floor" in info:
                self.logger.record("rollout/current_floor", info["floor"])
            if "hp_percent" in info:
                self.logger.record("rollout/hp_percent", info["hp_percent"])
        return True

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = StsEnv()
        log_dir = "./logs/"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, str(rank)))
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    print("=== Slay the Spire PPO Training Script (Overnight 60M) ===")
    
    # 1. Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("WARNING: CUDA not available. Training will be slow on CPU.")
        device = "cpu"

    # 2. Hyperparameters
    # 4060 Ti (8GB VRAM)
    num_cpu = 8 
    n_steps = 2048 # Standard for PPO
    batch_size = 512 # Increased batch size for efficiency
    n_epochs = 10
    total_timesteps = 60000000 # 60 Million Steps
    
    # Save Frequency: Every 2M steps
    # With 8 envs, we step 8 times per global step? No, SB3 counts total steps.
    # But callback frequency is per-env steps usually? 
    # CheckpointCallback: save_freq is "number of steps of the vectorized environment".
    # i.e. save_freq=1000 with 8 envs -> saves every 1000 calls to env.step(), which is 8000 frames.
    # User wants every 2M steps (frames).
    # So save_freq = 2,000,000 / num_cpu = 250,000.
    save_freq = 250000
    
    print(f"Initializing {num_cpu} parallel environments...")
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    # Eval Env for best_model
    eval_env = SubprocVecEnv([make_env(num_cpu)]) # Rank 8
    
    # 3. Model Definition
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256, 256], vf=[256, 256, 256])]
    )
    
    print("Initializing PPO Model...")
    
    # Find latest checkpoint
    checkpoints = glob.glob("./checkpoints/*.zip")
    latest_checkpoint = None
    if checkpoints:
        # Sort by modification time
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    
    model = None
    if latest_checkpoint:
        print(f"Resuming training from latest checkpoint: {latest_checkpoint}")
        try:
            model = PPO.load(
                latest_checkpoint,
                env=env,
                device=device,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                tensorboard_log="./tensorboard_logs/"
            )
            # Adjust num_timesteps if needed, but we rely on 'total_timesteps' in learn() being additional or absolute
            # SB3 PPO.load restores num_timesteps.
            remaining_timesteps = max(total_timesteps - model.num_timesteps, 1000000)
            print(f"Model loaded. Current timesteps: {model.num_timesteps}. Training for {remaining_timesteps} more.")
            
        except Exception as e:
            print(f"Failed to load checkpoint {latest_checkpoint}: {e}. Starting fresh.")
            latest_checkpoint = None
            
    if model is None:
        print("Starting fresh training.")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01, 
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            device=device,
            policy_kwargs=policy_kwargs
        )
        remaining_timesteps = total_timesteps

    # 4. Callbacks
    os.makedirs('./checkpoints/', exist_ok=True)
    os.makedirs('./models/', exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path='./checkpoints/',
        name_prefix='sts_ppo_60m'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=save_freq,
        deterministic=True,
        render=False
    )
    
    tb_callback = TensorboardCallback()
    
    print(f"Starting Training Loop. Target: {total_timesteps} steps.")
    print(f"Logging to ./tensorboard_logs/ and ./training_overnight.log")
    
    import traceback
    try:
        model.learn(
            total_timesteps=remaining_timesteps, 
            callback=[checkpoint_callback, eval_callback, tb_callback],
            progress_bar=True,
            reset_num_timesteps=False
        )
    except Exception as e:
        print(f"Training interrupted or failed:")
        traceback.print_exc()
        env.close()
        eval_env.close()
        raise e
        
    print("Training Finished.")
    model.save("sts_ppo_60m_final")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
