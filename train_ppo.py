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
    # Num Envs: 8 (Adjust based on CPU cores, usually logical_cores - 2)
    # n_steps: 2048 (Default PPO) -> Total buffer size = 8 * 2048 = 16,384 transitions per update.
    # batch_size: 1024 or 2048.
    
    num_cpu = 8 # Set to 8-12 as requested
    n_steps = 2048
    batch_size = 2048 
    n_epochs = 10
    total_timesteps = 100000 # 100k Steps
    
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
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # Encourage exploration
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device=device,
        policy_kwargs=policy_kwargs
    )
    
    # 4. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, # Every 10k steps
        save_path='./checkpoints/',
        name_prefix='sts_ppo'
    )
    
    tb_callback = TensorboardCallback()
    
    print(f"Starting Training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=[checkpoint_callback, tb_callback],
            progress_bar=True
        )
    except Exception as e:
        print(f"Training interrupted or failed: {e}")
        # Try to close envs
        env.close()
        raise e
        
    end_time = time.time()
    print(f"Training Finished in {end_time - start_time:.2f} seconds.")
    
    # Save Final Model
    model.save("sts_ppo_final")
    print("Model saved to sts_ppo_final.zip")
    
    env.close()

if __name__ == "__main__":
    main()
