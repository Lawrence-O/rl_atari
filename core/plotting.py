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
        """Plot the sigma parameter values in NoisyNet layers with scientific notation for small values
        
        Args:
            frames: Frame numbers
            weight_sigmas: Average sigma values for weights
            bias_sigmas: Average sigma values for biases (optional)
            save_name: Name to save plot as
        """
        plt.figure()
        
        # Plot with scientific notation if values are very small
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
        
        # Set y-scale to logarithmic to see small values clearly
        plt.yscale('log')  # Use log scale to see small values
        
        plt.ylabel("Sigma Value (log scale)")
        plt.title(f"NoisyNet sigma parameters - {self.env_name}")
        
        # Add scientific notation formatter for y-axis
        plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Add horizontal reference line for minimum value
        min_sigma = 1e-3  # Adjust this if your min_sigma value is different
        plt.axhline(y=min_sigma, color='r', linestyle='--', label=f"Min σ = {min_sigma:.1e}")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
        
        # Also create a version with linear scale for comparison
        plt.figure()
        plt.plot(frames, weight_sigmas, 'o-', label="Weight Sigma")
        if bias_sigmas:
            plt.plot(frames, bias_sigmas, 'o-', alpha=0.7, label="Bias Sigma")
        plt.legend()
        
        # Format x-axis
        if frames and max(frames) > 1000000:
            plt.xlabel("Frames (millions)")
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x/1000000:.1f}M')
        else:
            plt.xlabel("Frames")
        
        # Linear scale
        plt.ylabel("Sigma Value (linear scale)")
        plt.title(f"NoisyNet sigma parameters (linear) - {self.env_name}")
        
        # Scientific notation for small values
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        plt.axhline(y=min_sigma, color='r', linestyle='--', label=f"Min σ = {min_sigma:.1e}")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}_linear.png"))
        plt.close()
    
    def create_frame_plots(self, training_data, buffer_capacity):
        """Create frame-based plots from training data
        
        Args:
            training_data: Dictionary of training data with frame-based metrics
            buffer_capacity: Capacity of replay buffer
        """
        # Plot rewards over frames
        if "frames_vs_rewards" in training_data and training_data["frames_vs_rewards"]:
            frames, rewards = zip(*training_data["frames_vs_rewards"])
            
            plt.figure(figsize=(10, 6))
            plt.plot(frames, rewards, label="Training Rewards")
            plt.xlabel("Frames")
            plt.ylabel("Average Reward")
            plt.title(f"Training rewards - {self.env_name}")
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(self.plot_dir, "rewards_by_frame.png"))
            plt.close()
        
        # Plot evaluation rewards over frames
        if "frames_vs_eval_rewards" in training_data and training_data["frames_vs_eval_rewards"]:
            frames, rewards = zip(*training_data["frames_vs_eval_rewards"])
            self.plot_evaluation_rewards(frames, rewards)
        
        # Plot Q-values over frames
        if "frames_vs_q_values" in training_data and training_data["frames_vs_q_values"]:
            frames, q_values = zip(*training_data["frames_vs_q_values"])
            self.plot_q_values(frames, q_values)
        
        # Plot loss over frames
        if "loss" in training_data and training_data["loss"]:
            frames, losses = zip(*training_data["loss"])
            # Create a custom loss plot with frames on x-axis
            plt.figure(figsize=(10, 6))
            plt.plot(frames, losses)
            plt.xlabel("Frames")
            plt.ylabel("Loss")
            plt.yscale('log')
            plt.title(f"Training loss - {self.env_name}")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, "loss_by_frame.png"))
            plt.close()
        
        # Plot FPS over frames
        if "fps" in training_data and training_data["fps"]:
            frames, fps_values = zip(*training_data["fps"])
            plt.figure(figsize=(10, 6))
            plt.plot(frames, fps_values)
            plt.axhline(y=np.mean(fps_values), color='r', linestyle='--', 
                    label=f"Avg: {np.mean(fps_values):.1f} FPS")
            plt.xlabel("Frames")
            plt.ylabel("Frames Per Second")
            plt.title(f"Training speed - {self.env_name}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, "fps.png"))
            plt.close()
            
        # Plot memory usage if available
        if hasattr(buffer_capacity, '__len__'):
            buffer_size = len(buffer_capacity)
            max_size = buffer_capacity.maxlen
        else:
            buffer_size = buffer_capacity
            max_size = buffer_capacity
            
        # Simple buffer capacity chart
        plt.figure(figsize=(8, 6))
        plt.bar(['Used', 'Capacity'], [buffer_size, max_size])
        plt.title(f"Replay Buffer - {self.env_name}")
        plt.ylabel("Size")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "buffer_capacity.png"))
        plt.close()
    def plot_returns_distribution(self, returns, save_name="returns_distribution"):
        """Plot distribution of returns
        
        Args:
            returns: List of returns
            save_name: Name to save plot as
        """
        plt.figure()
        plt.hist(returns, bins=30, alpha=0.7, color='blue')
        plt.axvline(np.mean(returns), color='red', linestyle='dashed', linewidth=1)
        plt.xlabel("Return Value")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Returns - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()

    def plot_advantage_histogram(self, advantages, save_name="advantage_histogram"):
        """Plot histogram of advantage values
        
        Args:
            advantages: List of advantage values
            save_name: Name to save plot as
        """
        plt.figure()
        plt.hist(advantages, bins=30, alpha=0.7, color='green')
        plt.axvline(np.mean(advantages), color='red', linestyle='dashed', linewidth=1)
        plt.xlabel("Advantage Value")
        plt.ylabel("Frequency")
        plt.title(f"Advantage Distribution - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()

    def plot_gradient_norms(self, episodes, policy_grad_norms, value_grad_norms=None, save_name="gradient_norms"):
        """Plot gradient norms over training
        
        Args:
            episodes: List of episode numbers
            policy_grad_norms: List of policy gradient norms
            value_grad_norms: List of value function gradient norms (optional)
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(episodes, policy_grad_norms, label="Policy Gradient Norm", color='blue')
        
        if value_grad_norms:
            plt.plot(episodes, value_grad_norms, label="Value Gradient Norm", color='orange')
        
        plt.xlabel("Episode")
        plt.ylabel("Gradient Norm")
        plt.title(f"Parameter Update Magnitudes - {self.env_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()

    def plot_entropy(self, episodes, entropy_values, save_name="policy_entropy"):
        """Plot policy entropy over episodes
        
        Args:
            episodes: List of episode numbers
            entropy_values: List of entropy values
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(episodes, entropy_values, color='purple')
        plt.xlabel("Episode")
        plt.ylabel("Entropy")
        plt.title(f"Policy Entropy - {self.env_name}")
        plt.grid(alpha=0.3)
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
        if "beta" in training_data and training_data["beta"]:
            beta_x_axis = get_metric_x_axis(training_data["beta"])
            if beta_x_axis:
                self.plot_beta(beta_x_axis, training_data["beta"])
        
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
        # Advantage distribution - plot the latest batch of advantages
        if "advantage_values" in training_data and training_data["advantage_values"]:
            # Just plot most recent batch if it's a list of lists
            if isinstance(training_data["advantage_values"][0], list):
                recent_advantages = training_data["advantage_values"][-1]
            else:
                recent_advantages = training_data["advantage_values"][-1000:] if len(training_data["advantage_values"]) > 1000 else training_data["advantage_values"]
            
            self.plot_advantage_histogram(recent_advantages)
        
        # Gradient norms
        if "policy_grad_norms" in training_data and training_data["policy_grad_norms"]:
            episodes = list(range(len(training_data["policy_grad_norms"])))
            value_grad_norms = training_data.get("value_grad_norms", None)
            self.plot_gradient_norms(episodes, training_data["policy_grad_norms"], value_grad_norms)
        
        # Policy entropy over time
        if "entropy" in training_data and training_data["entropy"]:
            # Create x-axis based on frequency of entropy collection
            entropy_episodes = list(range(len(training_data["entropy"])))
            self.plot_entropy(entropy_episodes, training_data["entropy"])
        
        # Returns distribution - use the most recent batch of returns if available
        if "returns" in training_data and training_data["returns"]:
            # Handle both individual returns and lists of returns
            if isinstance(training_data["returns"][0], list):
                flat_returns = [r for batch in training_data["returns"] for r in batch]
                self.plot_returns_distribution(flat_returns)
            else:
                self.plot_returns_distribution(training_data["returns"])
    def plot_option_selection_frequency(self, option_frequencies, save_name="option_selection_frequency"):
        """Plot how frequently each option is selected
        
        Args:
            option_frequencies: Dictionary or list of option selection counts
            save_name: Name to save plot as
        """
        plt.figure()
        
        if isinstance(option_frequencies, dict):
            options = list(option_frequencies.keys())
            frequencies = list(option_frequencies.values())
        else:
            options = list(range(len(option_frequencies)))
            frequencies = option_frequencies
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(options)))
        bars = plt.bar(options, frequencies, color=colors)
        
        # Add value labels on bars
        for bar, freq in zip(bars, frequencies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(frequencies)*0.01,
                    f'{freq}', ha='center', va='bottom')
        
        plt.xlabel("Option ID")
        plt.ylabel("Selection Count")
        plt.title(f"Option Selection Frequency - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()

    def plot_option_duration_distribution(self, option_durations, save_name="option_duration_distribution"):
        """Plot distribution of option durations
        
        Args:
            option_durations: List of option durations or dict by option
            save_name: Name to save plot as
        """
        plt.figure()
        
        if isinstance(option_durations, dict):
            # Plot separate histograms for each option
            colors = plt.cm.viridis(np.linspace(0, 1, len(option_durations)))
            for i, (option_id, durations) in enumerate(option_durations.items()):
                plt.hist(durations, bins=20, alpha=0.6, label=f"Option {option_id}", 
                        color=colors[i], density=True)
            plt.legend()
        else:
            # Plot single histogram for all options
            plt.hist(option_durations, bins=30, alpha=0.7, color='blue', density=True)
        
        plt.xlabel("Option Duration (steps)")
        plt.ylabel("Density")
        plt.title(f"Option Duration Distribution - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()

    def plot_option_returns(self, option_returns, save_name="option_returns"):
        """Plot average returns per option over time
        
        Args:
            option_returns: Dict of {option_id: [returns over time]}
            save_name: Name to save plot as
        """
        plt.figure()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(option_returns)))
        
        for i, (option_id, returns) in enumerate(option_returns.items()):
            # Smooth the returns
            window = min(50, len(returns) // 5) if returns else 1
            if window > 1 and len(returns) > window:
                smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, window-1+len(smoothed)), smoothed, 
                        color=colors[i], label=f"Option {option_id}")
            else:
                plt.plot(returns, color=colors[i], label=f"Option {option_id}")
        
        plt.xlabel("Option Episode")
        plt.ylabel("Return")
        plt.title(f"Option Returns Over Time - {self.env_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()

    def plot_termination_probabilities(self, frames, termination_probs, save_name="termination_probabilities"):
        """Plot termination probabilities for each option over time
        
        Args:
            frames: Frame numbers
            termination_probs: Dict of {option_id: [termination_probs over time]}
            save_name: Name to save plot as
        """
        plt.figure()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(termination_probs)))
        
        for i, (option_id, probs) in enumerate(termination_probs.items()):
            plt.plot(frames, probs, color=colors[i], label=f"Option {option_id}")
        
        # Format x-axis for readability
        if frames and max(frames) > 1000000:
            plt.xlabel("Frames (millions)")
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x/1000000:.1f}M')
        else:
            plt.xlabel("Frames")
        
        plt.ylabel("Termination Probability")
        plt.title(f"Option Termination Probabilities - {self.env_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()

    def plot_meta_controller_q_values(self, frames, meta_q_values, save_name="meta_controller_q_values"):
        """Plot meta-controller Q-values over time
        
        Args:
            frames: Frame numbers
            meta_q_values: Meta-controller Q-values
            save_name: Name to save plot as
        """
        plt.figure()
        plt.plot(frames, meta_q_values, 'o-', color='red')
        
        # Format x-axis for readability
        if frames and max(frames) > 1000000:
            plt.xlabel("Frames (millions)")
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x/1000000:.1f}M')
        else:
            plt.xlabel("Frames")
        
        plt.ylabel("Average Meta Q-Value")
        plt.title(f"Meta-Controller Q-Values - {self.env_name}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()

    def plot_option_policy_losses(self, frames, option_losses, save_name="option_policy_losses"):
        """Plot training losses for each option policy
        
        Args:
            frames: Frame numbers
            option_losses: Dict of {option_id: [losses over time]}
            save_name: Name to save plot as
        """
        plt.figure()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(option_losses)))
        
        for i, (option_id, losses) in enumerate(option_losses.items()):
            # Filter out None values
            valid_frames = []
            valid_losses = []
            for f, loss in zip(frames, losses):
                if loss is not None:
                    valid_frames.append(f)
                    valid_losses.append(loss)
            
            if valid_losses:
                plt.plot(valid_frames, valid_losses, color=colors[i], 
                        label=f"Option {option_id}", alpha=0.7)
        
        # Format x-axis for readability
        if frames and max(frames) > 1000000:
            plt.xlabel("Frames (millions)")
            plt.gca().xaxis.set_major_formatter(lambda x, pos: f'{x/1000000:.1f}M')
        else:
            plt.xlabel("Frames")
        
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.title(f"Option Policy Losses - {self.env_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()
    def plot_hierarchical_learning_summary(self, training_data, save_name="hierarchical_summary"):
        """Create a comprehensive summary plot for hierarchical learning
        
        Args:
            training_data: Training data dictionary
            save_name: Name to save plot as
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Hierarchical Learning Summary - {self.env_name}", fontsize=16)
        
        # 1. Option Selection Frequency
        if "option_frequencies" in training_data:
            ax = axes[0, 0]
            option_freq = training_data["option_frequencies"]
            if isinstance(option_freq, dict):
                options = list(option_freq.keys())
                frequencies = list(option_freq.values())
            else:
                options = list(range(len(option_freq)))
                frequencies = option_freq
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(options)))
            bars = ax.bar(options, frequencies, color=colors)
            ax.set_xlabel("Option ID")
            ax.set_ylabel("Selection Count")
            ax.set_title("Option Selection Frequency")
            
            # Add value labels
            for bar, freq in zip(bars, frequencies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(frequencies)*0.01,
                    f'{freq}', ha='center', va='bottom', fontsize=8)
        
        # 2. Meta-Controller vs Option Losses
        if "meta_loss" in training_data and "option_losses" in training_data:
            ax = axes[0, 1]
            if training_data["meta_loss"]:
                meta_frames = list(range(len(training_data["meta_loss"])))
                ax.plot(meta_frames, training_data["meta_loss"], label="Meta-Controller", color='red')
            
            # Average option losses
            if training_data["option_losses"]:
                option_frames = list(range(len(training_data["option_losses"])))
                # Calculate average loss across options (excluding None values)
                avg_option_losses = []
                for losses_at_frame in training_data["option_losses"]:
                    valid_losses = [l for l in losses_at_frame if l is not None]
                    if valid_losses:
                        avg_option_losses.append(np.mean(valid_losses))
                    else:
                        avg_option_losses.append(None)
                
                valid_frames = [f for f, l in zip(option_frames, avg_option_losses) if l is not None]
                valid_losses = [l for l in avg_option_losses if l is not None]
                
                if valid_losses:
                    ax.plot(valid_frames, valid_losses, label="Avg Option Loss", color='blue')
            
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Loss")
            ax.set_title("Meta vs Option Losses")
            ax.legend()
            ax.set_yscale('log')
            ax.grid(alpha=0.3)
        
        # 3. Option Duration Trends
        if "option_durations" in training_data:
            ax = axes[0, 2]
            durations = training_data["option_durations"]
            if isinstance(durations, dict):
                colors = plt.cm.viridis(np.linspace(0, 1, len(durations)))
                for i, (option_id, durs) in enumerate(durations.items()):
                    # Plot moving average of durations
                    if len(durs) > 10:
                        window = min(50, len(durs) // 5)
                        smoothed = np.convolve(durs, np.ones(window)/window, mode='valid')
                        ax.plot(range(window-1, window-1+len(smoothed)), smoothed, 
                            color=colors[i], label=f"Option {option_id}")
                    else:
                        ax.plot(durs, color=colors[i], label=f"Option {option_id}")
            else:
                # Single plot for all durations
                ax.plot(durations, color='blue')
            
            ax.set_xlabel("Option Episode")
            ax.set_ylabel("Duration (steps)")
            ax.set_title("Option Duration Trends")
            if isinstance(durations, dict):
                ax.legend()
            ax.grid(alpha=0.3)
        
        # 4. Termination Probability Evolution
        if "termination_probabilities" in training_data:
            ax = axes[1, 0]
            term_probs = training_data["termination_probabilities"]
            frames = training_data.get("eval_frames", list(range(len(list(term_probs.values())[0]))))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(term_probs)))
            for i, (option_id, probs) in enumerate(term_probs.items()):
                ax.plot(frames[:len(probs)], probs, color=colors[i], label=f"Option {option_id}")
            
            ax.set_xlabel("Frames")
            ax.set_ylabel("Termination Probability")
            ax.set_title("Termination Probabilities")
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 5. Option Returns Comparison
        if "option_returns" in training_data:
            ax = axes[1, 1]
            option_rets = training_data["option_returns"]
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(option_rets)))
            for i, (option_id, returns) in enumerate(option_rets.items()):
                # Plot smoothed returns
                if len(returns) > 10:
                    window = min(20, len(returns) // 3)
                    smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
                    ax.plot(range(window-1, window-1+len(smoothed)), smoothed, 
                        color=colors[i], label=f"Option {option_id}")
                else:
                    ax.plot(returns, color=colors[i], label=f"Option {option_id}")
            
            ax.set_xlabel("Option Episode")
            ax.set_ylabel("Return")
            ax.set_title("Option Returns")
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 6. Overall Performance Comparison
        ax = axes[1, 2]
        if "eval_rewards" in training_data and "eval_frames" in training_data:
            eval_frames = training_data["eval_frames"]
            eval_rewards = training_data["eval_rewards"]
            ax.plot(eval_frames, eval_rewards, 'o-', color='green', label="Hierarchical Agent")
            
            # If we have baseline data, plot it too
            if "baseline_rewards" in training_data:
                ax.plot(eval_frames, training_data["baseline_rewards"], 'o-', 
                    color='orange', alpha=0.7, label="Baseline")
            
            ax.set_xlabel("Frames")
            ax.set_ylabel("Evaluation Reward")
            ax.set_title("Overall Performance")
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()

    def plot_option_transition_matrix(self, transition_matrix, save_name="option_transition_matrix"):
        """Plot option-to-option transition matrix as heatmap
        
        Args:
            transition_matrix: 2D array showing transitions between options
            save_name: Name to save plot as
        """
        plt.figure(figsize=(8, 6))
        
        # Normalize rows to show transition probabilities
        transition_probs = transition_matrix / (transition_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        sns.heatmap(transition_probs, annot=True, fmt='.3f', cmap='viridis', 
                xticklabels=[f"Option {i}" for i in range(transition_matrix.shape[1])],
                yticklabels=[f"Option {i}" for i in range(transition_matrix.shape[0])])
        
        plt.xlabel("Next Option")
        plt.ylabel("Current Option")
        plt.title(f"Option Transition Probabilities - {self.env_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{save_name}.png"))
        plt.close()

    def create_hierarchical_plots(self, training_data):
        """Create all hierarchical learning plots
        
        Args:
            training_data: Dictionary containing hierarchical training data
        """
        # Option selection frequency
        if "option_frequencies" in training_data:
            self.plot_option_selection_frequency(training_data["option_frequencies"])
        
        # Option duration distribution
        if "option_durations" in training_data:
            self.plot_option_duration_distribution(training_data["option_durations"])
        
        # Option returns over time
        if "option_returns" in training_data:
            self.plot_option_returns(training_data["option_returns"])
        
        # Termination probabilities
        if "termination_probabilities" in training_data:
            frames = training_data.get("eval_frames", 
                                    list(range(len(list(training_data["termination_probabilities"].values())[0]))))
            self.plot_termination_probabilities(frames, training_data["termination_probabilities"])
        
        # Meta-controller Q-values
        if "meta_q_values" in training_data:
            frames = training_data.get("eval_frames", list(range(len(training_data["meta_q_values"]))))
            self.plot_meta_controller_q_values(frames, training_data["meta_q_values"])
        
        # Option policy losses
        if "option_losses" in training_data:
            frames = list(range(len(training_data["option_losses"])))
            # Reorganize data by option
            num_options = len(training_data["option_losses"][0]) if training_data["option_losses"] else 0
            option_losses_by_id = {}
            for option_id in range(num_options):
                option_losses_by_id[option_id] = [
                    losses[option_id] if losses[option_id] is not None else None 
                    for losses in training_data["option_losses"]
                ]
            self.plot_option_policy_losses(frames, option_losses_by_id)
        
        # Option transition matrix
        if "option_transition_matrix" in training_data:
            self.plot_option_transition_matrix(training_data["option_transition_matrix"])
        
        # Comprehensive summary
        self.plot_hierarchical_learning_summary(training_data)