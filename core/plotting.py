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
        sns.set_theme(style="darkgrid")
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
                            frames: List[int],
                            rewards: List[float],
                            save_name: str = "eval_rewards"):
        """Plot evaluation rewards vs frames
        
        Args:
            frames: Frame numbers
            rewards: Evaluation rewards
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(frames, rewards, 'o-')
        
        # Format x-axis for readability
        if frames and max(frames) > 1000000:
            plt.xlabel("Frames (millions)")
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x/1000000:.1f}M')
        else:
            plt.xlabel("Frames")
            
        plt.ylabel("Evaluation Reward")
        plt.title(f"Evaluation rewards - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_q_values(self, 
                     frames: List[int], 
                     q_values: List[float], 
                     save_name: str = "q_values"):
        """Plot Q-values
        
        Args:
            frames: Frame numbers
            q_values: Q-values
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(frames, q_values, 'o-')
        
        # Format x-axis for readability
        if frames and max(frames) > 1000000:
            plt.xlabel("Frames (millions)")
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x/1000000:.1f}M')
        else:
            plt.xlabel("Frames")
            
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
        plt.xlabel("Update Step")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.title(f"Training loss - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_td_errors(self, 
                      frames: List[int], 
                      mean_td_errors: List[float],
                      max_td_errors: Optional[List[float]] = None,
                      save_name: str = "td_errors"):
        """Plot TD errors
        
        Args:
            frames: Frame numbers
            mean_td_errors: Mean TD errors
            max_td_errors: Maximum TD errors (optional)
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(frames, mean_td_errors, label="Mean TD Error")
        
        if max_td_errors:
            plt.plot(frames, max_td_errors, alpha=0.5, label="Max TD Error")
            plt.legend()
        
        # Format x-axis for readability
        if frames and max(frames) > 1000000:
            plt.xlabel("Frames (millions)")
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x/1000000:.1f}M')
        else:
            plt.xlabel("Frames")
            
        plt.ylabel("TD Error")
        plt.title(f"TD errors - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_memory_usage(self, 
                         frames: List[int], 
                         memory_size: List[int],
                         capacity: int,
                         save_name: str = "memory_usage"):
        """Plot memory buffer usage
        
        Args:
            frames: Frame numbers
            memory_size: Memory sizes
            capacity: Total buffer capacity
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(frames, memory_size)
        plt.axhline(y=capacity, color='r', linestyle='--', label="Capacity")
        
        # Format x-axis for readability
        if frames and max(frames) > 1000000:
            plt.xlabel("Frames (millions)")
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x/1000000:.1f}M')
        else:
            plt.xlabel("Frames")
            
        plt.ylabel("Buffer Size")
        plt.title(f"Replay buffer usage - {self.env_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_per_weights(self, 
                        frames: List[int], 
                        weight_variance: List[float],
                        save_name: str = "per_weights"):
        """Plot PER weight variance
        
        Args:
            frames: Frame numbers
            weight_variance: Weight variance values
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(frames, weight_variance)
        
        # Format x-axis for readability
        if frames and max(frames) > 1000000:
            plt.xlabel("Frames (millions)")
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x/1000000:.1f}M')
        else:
            plt.xlabel("Frames")
            
        plt.ylabel("Weight Variance")
        plt.title(f"PER weight variance - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_beta(self, 
                 frames: List[int], 
                 beta_values: List[float],
                 save_name: str = "beta"):
        """Plot beta annealing
        
        Args:
            frames: Frame numbers
            beta_values: Beta values
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(frames, beta_values)
        plt.axhline(y=1.0, color='r', linestyle='--', label="Target")
        
        # Format x-axis for readability
        if frames and max(frames) > 1000000:
            plt.xlabel("Frames (millions)")
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x/1000000:.1f}M')
        else:
            plt.xlabel("Frames")
            
        plt.ylabel("Beta")
        plt.title(f"PER beta annealing - {self.env_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_network_distance(self, 
                             frames: List[int], 
                             distances: List[float],
                             save_name: str = "network_distance"):
        """Plot network distance
        
        Args:
            frames: Frame numbers
            distances: Network distances
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(frames, distances, 'o-')
        
        # Format x-axis for readability
        if frames and max(frames) > 1000000:
            plt.xlabel("Frames (millions)")
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x/1000000:.1f}M')
        else:
            plt.xlabel("Frames")
            
        plt.ylabel("L1 Distance")
        plt.title(f"Policy-Target network distance - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_epsilon(self, 
                    frames: List[int], 
                    epsilon_values: List[float],
                    save_name: str = "epsilon"):
        """Plot epsilon decay
        
        Args:
            frames: Frame numbers
            epsilon_values: Epsilon values
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(frames, epsilon_values)
        
        # Format x-axis for readability
        if frames and max(frames) > 1000000:
            plt.xlabel("Frames (millions)")
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x/1000000:.1f}M')
        else:
            plt.xlabel("Frames")
            
        plt.ylabel("Epsilon")
        plt.title(f"Exploration rate (epsilon) - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
        
    def plot_noise_magnitude(self,
                          frames: List[int],
                          noise_magnitudes: List[float],
                          save_name: str = "noise_magnitude"):
        """Plot the magnitude of noise in NoisyNet layers
        
        Args:
            frames: Frame numbers
            noise_magnitudes: Average magnitude of noise
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(frames, noise_magnitudes, 'o-')
        
        # Format x-axis for readability
        if frames and max(frames) > 1000000:
            plt.xlabel("Frames (millions)")
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x/1000000:.1f}M')
        else:
            plt.xlabel("Frames")
            
        plt.ylabel("Noise Magnitude")
        plt.title(f"NoisyNet noise magnitude - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    
    def plot_sigma_params(self,
                        frames: List[int],
                        weight_sigmas: List[float],
                        bias_sigmas: List[float] = None,
                        save_name: str = "sigma_params"):
        """Plot the sigma parameter values in NoisyNet layers
        
        Args:
            frames: Frame numbers
            weight_sigmas: Average sigma values for weights
            bias_sigmas: Average sigma values for biases (optional)
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(frames, weight_sigmas, 'o-', label="Weight Sigma")
        
        if bias_sigmas:
            plt.plot(frames, bias_sigmas, 'o-', alpha=0.7, label="Bias Sigma")
            plt.legend()
            
        # Format x-axis for readability
        if frames and max(frames) > 1000000:
            plt.xlabel("Frames (millions)")
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x/1000000:.1f}M')
        else:
            plt.xlabel("Frames")
            
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
        # Get the x-axis values (use frames or fallback to episodes)
        eval_x_axis = training_data.get("eval_frames", training_data.get("eval_episodes", []))

        # Basic plots
        if "episode_rewards" in training_data and training_data["episode_rewards"]:
            self.plot_episode_rewards(training_data["episode_rewards"])
            self.plot_smoothed_rewards(training_data["episode_rewards"])
        
        # Evaluation rewards - uses eval frames
        if "eval_rewards" in training_data and training_data["eval_rewards"] and eval_x_axis:
            self.plot_evaluation_rewards(eval_x_axis, training_data["eval_rewards"])
        
        # Training loss - uses update steps as x-axis
        if "training_losses" in training_data and training_data["training_losses"]:
            self.plot_loss(training_data["training_losses"])
        
        # Q-values - uses eval frames
        if "q_values" in training_data and training_data["q_values"] and eval_x_axis:
            self.plot_q_values(eval_x_axis, training_data["q_values"])

        # Generate frames for metrics collected more frequently
        # Generate evenly spaced x-axis values for metrics that don't align with eval points
        def get_metric_x_axis(metric_values):
            if not metric_values:
                return []
            
            # For metrics that are collected more frequently than eval points, 
            # create a synthetic x-axis ranging from 0 to max_frames
            max_frames = max(eval_x_axis) if eval_x_axis else 0
            return np.linspace(0, max_frames, len(metric_values)).tolist()
        
        # PER metrics - create synthetic x-axis
        if "weight_variance" in training_data and training_data["weight_variance"]:
            per_x_axis = get_metric_x_axis(training_data["weight_variance"])
            if per_x_axis:
                self.plot_per_weights(per_x_axis, training_data["weight_variance"])
        
        # Beta annealing - create synthetic x-axis
        if "beta_values" in training_data and training_data["beta_values"]:
            beta_x_axis = get_metric_x_axis(training_data["beta_values"])
            if beta_x_axis:
                self.plot_beta(beta_x_axis, training_data["beta_values"])
        
        # TD errors - create synthetic x-axis
        if "mean_td_errors" in training_data and training_data["mean_td_errors"]:
            td_x_axis = get_metric_x_axis(training_data["mean_td_errors"])
            if td_x_axis:
                max_td_errors = training_data.get("max_td_errors", None)
                self.plot_td_errors(td_x_axis, training_data["mean_td_errors"], max_td_errors)
        
        # Memory usage - create synthetic x-axis
        if "memory_sizes" in training_data and training_data["memory_sizes"]:
            memory_x_axis = get_metric_x_axis(training_data["memory_sizes"])
            if memory_x_axis:
                self.plot_memory_usage(memory_x_axis, training_data["memory_sizes"], buffer_capacity)
        
        # Network distance - create synthetic x-axis
        if "network_distances" in training_data and training_data["network_distances"]:
            net_x_axis = get_metric_x_axis(training_data["network_distances"])
            if net_x_axis:
                self.plot_network_distance(net_x_axis, training_data["network_distances"])
        
        # Epsilon decay - create synthetic x-axis
        if "epsilon_values" in training_data and training_data["epsilon_values"]:
            eps_x_axis = get_metric_x_axis(training_data["epsilon_values"])
            if eps_x_axis:
                self.plot_epsilon(eps_x_axis, training_data["epsilon_values"])
        
        # NoisyNet metrics - create synthetic x-axis
        if "noise_magnitudes" in training_data and training_data["noise_magnitudes"]:
            noise_x_axis = get_metric_x_axis(training_data["noise_magnitudes"])
            if noise_x_axis:
                self.plot_noise_magnitude(noise_x_axis, training_data["noise_magnitudes"])
        elif "noise_magnitude" in training_data and training_data["noise_magnitude"]:
            noise_x_axis = get_metric_x_axis(training_data["noise_magnitude"])
            if noise_x_axis:
                self.plot_noise_magnitude(noise_x_axis, training_data["noise_magnitude"])
        
        # Sigma parameters - create synthetic x-axis
        if "weight_sigmas" in training_data and training_data["weight_sigmas"]:
            sigma_x_axis = get_metric_x_axis(training_data["weight_sigmas"])
            if sigma_x_axis:
                bias_sigmas = training_data.get("bias_sigmas", None)
                self.plot_sigma_params(sigma_x_axis, training_data["weight_sigmas"], bias_sigmas)
        elif "weight_sigma" in training_data and training_data["weight_sigma"]:
            sigma_x_axis = get_metric_x_axis(training_data["weight_sigma"])
            if sigma_x_axis:
                bias_sigma = training_data.get("bias_sigma", None)
                self.plot_sigma_params(sigma_x_axis, training_data["weight_sigma"], bias_sigma)