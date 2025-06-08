import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import yaml
import os
from datetime import datetime

def set_seed(seed):
    """Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_experiment_dir(base_dir="experiments", experiment_name=None):
    """Create a directory for experiment results
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to experiment directory
    """
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir

def plot_learning_curve(rewards, avg_window=100, title="Learning Curve", save_path=None):
    """Plot learning curve from episode rewards
    
    Args:
        rewards: List of episode rewards
        avg_window: Window size for moving average
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.6, label="Rewards")
    
    # Plot moving average
    if len(rewards) >= avg_window:
        moving_avg = np.convolve(rewards, np.ones(avg_window) / avg_window, mode='valid')
        plt.plot(range(avg_window-1, len(rewards)), moving_avg, 'r-', 
                label=f"{avg_window}-episode Moving Average")
    
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

class Timer:
    """Simple timer for measuring elapsed time"""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer and return elapsed time"""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self.elapsed
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, *args):
        self.stop()