import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Dict, Any, Optional

class RLPlotter:
    """Class for plotting RL training metrics"""
    
    def __init__(self, plot_dir: str, env_name: str = "Unknown"):
        """Initialize the plotter
        
        Args:
            plot_dir: Directory to save plots
            env_name: Name of the environment (for plot titles)
        """
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Clean env name for titles
        self.env_name = env_name.split('/')[-1] if '/' in env_name else env_name
        
        # Set plot style
        sns.set(style="darkgrid")
        plt.rcParams["figure.figsize"] = (10, 6)
    
    def plot_episode_rewards(self, rewards: List[float], save_name: str = "rewards"):
        """Plot episode rewards
        
        Args:
            rewards: List of episode rewards
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Training rewards - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_smoothed_rewards(self, rewards: List[float], save_name: str = "rewards_smoothed"):
        """Plot smoothed episode rewards
        
        Args:
            rewards: List of episode rewards
            save_name: Name to save plot as
        """
        plt.figure()
        window = min(50, len(rewards) // 5) if rewards else 1
        if window > 1 and len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(rewards, alpha=0.3, color='blue', label="Raw")
            plt.plot(range(window-1, window-1+len(smoothed)), smoothed, color='blue', label="Smoothed")
            plt.legend()
        else:
            plt.plot(rewards, color='blue')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Smoothed rewards - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_evaluation_rewards(self, 
                            frames: List[int],  # Changed from episodes to frames
                            rewards: List[float],
                            save_name: str = "eval_rewards"):
        """Plot evaluation rewards vs frames"""
        plt.figure()
        plt.plot(frames, rewards, 'o-')
        plt.xlabel("Frames (millions)")
        plt.ylabel("Evaluation Reward")
        plt.title(f"Evaluation rewards - {self.env_name}")
        
        # Format x-axis to show millions of frames
        if frames and frames[-1] > 1000000:
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{int(x/1000000)}M')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_q_values(self, 
                     episodes: List[int], 
                     q_values: List[float], 
                     save_name: str = "q_values"):
        """Plot Q-values
        
        Args:
            episodes: Episode numbers
            q_values: Q-values
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(episodes, q_values, 'o-')
        plt.xlabel("Episode")
        plt.ylabel("Average Q-Value")
        plt.title(f"Q-value evolution - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_loss(self, losses: List[float], save_name: str = "loss"):
        """Plot training loss
        
        Args:
            losses: List of training losses
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(losses)
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.title(f"Training loss - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_td_errors(self, 
                      episodes: List[int], 
                      mean_td_errors: List[float],
                      max_td_errors: Optional[List[float]] = None,
                      save_name: str = "td_errors"):
        """Plot TD errors
        
        Args:
            episodes: Episode numbers
            mean_td_errors: Mean TD errors
            max_td_errors: Maximum TD errors (optional)
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(episodes, mean_td_errors, label="Mean TD Error")
        
        if max_td_errors:
            plt.plot(episodes, max_td_errors, alpha=0.5, label="Max TD Error")
            plt.legend()
            
        plt.xlabel("Episode")
        plt.ylabel("TD Error")
        plt.title(f"TD errors - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_memory_usage(self, 
                         episodes: List[int], 
                         memory_size: List[int],
                         capacity: int,
                         save_name: str = "memory_usage"):
        """Plot memory buffer usage
        
        Args:
            episodes: Episode numbers
            memory_size: Memory sizes
            capacity: Total buffer capacity
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(episodes, memory_size)
        plt.axhline(y=capacity, color='r', linestyle='--', label="Capacity")
        plt.xlabel("Episode")
        plt.ylabel("Buffer Size")
        plt.title(f"Replay buffer usage - {self.env_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_per_weights(self, 
                        episodes: List[int], 
                        weight_variance: List[float],
                        save_name: str = "per_weights"):
        """Plot PER weight variance
        
        Args:
            episodes: Episode numbers
            weight_variance: Weight variance values
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(episodes, weight_variance)
        plt.xlabel("Episode")
        plt.ylabel("Weight Variance")
        plt.title(f"PER weight variance - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_beta(self, 
                 episodes: List[int], 
                 beta_values: List[float],
                 save_name: str = "beta"):
        """Plot beta annealing
        
        Args:
            episodes: Episode numbers
            beta_values: Beta values
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(episodes, beta_values)
        plt.axhline(y=1.0, color='r', linestyle='--', label="Target")
        plt.xlabel("Episode")
        plt.ylabel("Beta")
        plt.title(f"PER beta annealing - {self.env_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_network_distance(self, 
                             episodes: List[int], 
                             distances: List[float],
                             save_name: str = "network_distance"):
        """Plot network distance
        
        Args:
            episodes: Episode numbers
            distances: Network distances
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(episodes, distances, 'o-')
        plt.xlabel("Episode")
        plt.ylabel("L1 Distance")
        plt.title(f"Policy-Target network distance - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_epsilon(self, 
                    episodes: List[int], 
                    epsilon_values: List[float],
                    save_name: str = "epsilon"):
        """Plot epsilon decay
        
        Args:
            episodes: Episode numbers
            epsilon_values: Epsilon values
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(episodes, epsilon_values)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title(f"Exploration rate (epsilon) - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    def plot_noise_magnitude(self,
                        episodes: List[int],
                        noise_magnitudes: List[float],
                        save_name: str = "noise_magnitude"):
        """Plot the magnitude of noise in NoisyNet layers
        
        Args:
            episodes: Episode numbers
            noise_magnitudes: Average magnitude of noise
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(episodes, noise_magnitudes, 'o-')
        plt.xlabel("Episode")
        plt.ylabel("Noise Magnitude")
        plt.title(f"NoisyNet noise magnitude - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_sigma_params(self,
                        episodes: List[int],
                        weight_sigmas: List[float],
                        bias_sigmas: List[float] = None,
                        save_name: str = "sigma_params"):
        """Plot the sigma parameter values in NoisyNet layers
        
        Args:
            episodes: Episode numbers
            weight_sigmas: Average sigma values for weights
            bias_sigmas: Average sigma values for biases (optional)
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(episodes, weight_sigmas, 'o-', label="Weight Sigma")
        
        if bias_sigmas:
            plt.plot(episodes, bias_sigmas, 'o-', alpha=0.7, label="Bias Sigma")
            plt.legend()
            
        plt.xlabel("Episode")
        plt.ylabel("Sigma Value")
        plt.title(f"NoisyNet sigma parameters - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def create_all_plots(self, training_data: Dict[str, Any], buffer_capacity: int):
        """Create all available plots from training data
        
        Args:
            training_data: Dictionary of training data
            buffer_capacity: Capacity of replay buffer
        """
        # Basic plots
        if "episode_rewards" in training_data and training_data["episode_rewards"]:
            self.plot_episode_rewards(training_data["episode_rewards"])
            self.plot_smoothed_rewards(training_data["episode_rewards"])
        
        if "eval_episodes" in training_data and "eval_rewards" in training_data:
            if training_data["eval_episodes"] and training_data["eval_rewards"]:
                self.plot_evaluation_rewards(
                    training_data["eval_episodes"], 
                    training_data["eval_rewards"]
                )
        
        if "training_losses" in training_data and training_data["training_losses"]:
            self.plot_loss(training_data["training_losses"])
        
        # Advanced metrics
        if "eval_episodes" in training_data and "q_values" in training_data:
            if training_data["eval_episodes"] and training_data["q_values"]:
                self.plot_q_values(
                    training_data["eval_episodes"], 
                    training_data["q_values"]
                )
        
        # PER metrics
        if "eval_episodes" in training_data and "weight_variance" in training_data:
            if training_data["eval_episodes"] and training_data["weight_variance"]:
                self.plot_per_weights(
                    training_data["eval_episodes"], 
                    training_data["weight_variance"]
                )
        
        # Beta annealing
        if "eval_episodes" in training_data and "beta_values" in training_data:
            if training_data["eval_episodes"] and training_data["beta_values"]:
                self.plot_beta(
                    training_data["eval_episodes"], 
                    training_data["beta_values"]
                )
        
        # TD errors
        if "eval_episodes" in training_data and "mean_td_errors" in training_data:
            if (training_data["eval_episodes"] and 
                training_data["mean_td_errors"]):
                max_td_errors = (training_data.get("max_td_errors") 
                               if "max_td_errors" in training_data 
                               else None)
                self.plot_td_errors(
                    training_data["eval_episodes"], 
                    training_data["mean_td_errors"],
                    max_td_errors
                )
        
        # Memory usage
        if "eval_episodes" in training_data and "memory_sizes" in training_data:
            if training_data["eval_episodes"] and training_data["memory_sizes"]:
                self.plot_memory_usage(
                    training_data["eval_episodes"], 
                    training_data["memory_sizes"],
                    buffer_capacity
                )
        
        # Network distance
        if "eval_episodes" in training_data and "network_distances" in training_data:
            if training_data["eval_episodes"] and training_data["network_distances"]:
                self.plot_network_distance(
                    training_data["eval_episodes"], 
                    training_data["network_distances"]
                )
        
        # Epsilon decay
        if "eval_episodes" in training_data and "epsilon_values" in training_data:
            if training_data["eval_episodes"] and training_data["epsilon_values"]:
                self.plot_epsilon(
                    training_data["eval_episodes"], 
                    training_data["epsilon_values"]
                )
        # NoisyNet metrics
        if "eval_episodes" in training_data and "noise_magnitudes" in training_data:
            if training_data["eval_episodes"] and training_data["noise_magnitudes"]:
                self.plot_noise_magnitude(
                    training_data["eval_episodes"], 
                    training_data["noise_magnitudes"]
                )
        if "eval_episodes" in training_data and "weight_sigmas" in training_data:
            if training_data["eval_episodes"] and training_data["weight_sigmas"]:
                bias_sigmas = (training_data.get("bias_sigmas") 
                             if "bias_sigmas" in training_data 
                             else None)
                self.plot_sigma_params(
                    training_data["eval_episodes"], 
                    training_data["weight_sigmas"],
                    bias_sigmas
                )