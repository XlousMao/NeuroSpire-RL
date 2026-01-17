import os
import time
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# Import our custom environment
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
            if "episode" in info: # Standard Monitor wrapper key
                # Monitor wrapper usually puts 'r', 'l', 't' in 'episode' dict
                pass
        return True

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        # Create environment
        env = StsEnv()
        # Wrap the environment with Monitor to log stats (reward, length)
        log_dir = "./logs/"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, str(rank)))
        # Reseed
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    print("=== Slay the Spire PPO Training Script ===")
    
    # 1. Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB / {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
        device = "cuda"
    else:
        print("WARNING: CUDA not available. Training will be slow on CPU.")
        device = "cpu"

    # 2. Hyperparameters for High-Throughput Training
    # 4060 Ti has 16GB VRAM, so we can use large batch sizes.
    # Num Envs: 12 (Maximize CPU/GPU Throughput)
    # n_steps: 1024 -> Buffer size = 12 * 1024 = 12,288
    # batch_size: 256
    
    num_cpu = 8 # Reduce from 12 to 8 to avoid BrokenPipeError/Deadlock
    n_steps = 1024
    batch_size = 256
    n_epochs = 10
    total_timesteps = 1000000 # 1 Million Steps
    
    print(f"Initializing {num_cpu} parallel environments...")
    
    # Create the vectorized environment
    # On Windows, SubprocVecEnv requires standard entry point protection
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    # 3. Model Definition
    # MlpPolicy with custom network architecture
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256, 256], vf=[256, 256, 256])]
    )
    
    print("Initializing PPO Model...")
    
    # Check for existing checkpoint to resume
    checkpoint_path = "./checkpoints/sts_ppo_1m_800000_steps.zip"
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = PPO.load(
            checkpoint_path,
            env=env,
            device=device,
            # We must pass custom objects if they changed, but PPO usually saves them.
            # However, we want to ensure we keep our new hyperparameters if we are finetuning.
            # load() overwrites params. We can force them back if needed, but resuming usually implies continuing same config.
            # If we want to change n_steps/batch_size, we might need to re-init and load weights only.
            # But for simplicity and stability, let's trust the load or re-init if load fails.
            # Actually, PPO.load() will load the saved hyperparameters.
            # If we changed n_steps/batch_size in code, loading old model might have old params.
            # Let's check if we can override.
            # kwargs can override.
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            ent_coef=0.01,
            clip_range=0.2,
            tensorboard_log="./tensorboard_logs/"
        )
        # Reset num_timesteps to reflect resumed state? 
        # model.num_timesteps is loaded. 
        # total_timesteps in learn() is "total timesteps to train for" (in addition? or total?)
        # SB3 learn(total_timesteps) runs for that many *additional* steps if reset_num_timesteps=False.
        # But we want to complete the 1M.
        # If model has 800k, we want 200k more.
        remaining_timesteps = total_timesteps - model.num_timesteps
        if remaining_timesteps < 0: remaining_timesteps = 200000 # Fallback
        print(f"Resuming for remaining {remaining_timesteps} steps...")
        
    else:
        print("No checkpoint found. Starting fresh (or checkpoint name mismatch).")
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
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, # Every 100k steps
        save_path='./checkpoints/',
        name_prefix='sts_ppo_1m'
    )
    
    tb_callback = TensorboardCallback()
    
    print(f"Starting Training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=remaining_timesteps, 
            callback=[checkpoint_callback, tb_callback],
            progress_bar=True,
            reset_num_timesteps=False
        )
    except Exception as e:
        print(f"Training interrupted or failed: {e}")
        # Try to close envs
        env.close()
        raise e
        
    end_time = time.time()
    print(f"Training Finished in {end_time - start_time:.2f} seconds.")
    
    # Save Final Model
    model.save("sts_ppo_1m_final")
    print("Model saved to sts_ppo_1m_final.zip")
    
    # Also save to build/Release for convenience if running from root
    import shutil
    try:
        if os.path.exists("build/Release"):
             shutil.copy("sts_ppo_1m_final.zip", "build/Release/sts_ppo_1m_final.zip")
             print("Model copied to build/Release/sts_ppo_1m_final.zip")
    except:
        pass
    
    env.close()

if __name__ == "__main__":
    main()
