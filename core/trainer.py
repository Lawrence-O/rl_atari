import os
import time
import json
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from gymnasium.wrappers import RecordVideo
from PIL import Image

from core.environment import AtariWrapper
from core.utils import set_seed
from core.plotting import RLPlotter  # New import for plotting


class Timer:
    """Simple timer for tracking elapsed time."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.time()
        self.elapsed = 0
    
    def update(self):
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time


class RLTrainer:
    """Generic trainer for reinforcement learning agents."""
    
    def __init__(self, agent_class, config, experiment_name=None, experiment_dir=None):
        """Initialize trainer.
        
        Args:
            agent_class: Agent class to train
            config: Configuration dictionary
            experiment_name: Name of experiment
            experiment_dir: Directory to save results
        """
        self.config = config
        self.agent_class = agent_class
        
        # Set up experiment directory
        if experiment_dir is None:
            base_dir = "experiments/results"
            experiment_name = experiment_name or f"{agent_class.__name__}_{config['env_name'].split('/')[-1].lower()}"
            self.experiment_dir = self._create_experiment_dir(base_dir, experiment_name)
        else:
            self.experiment_dir = experiment_dir
            
        # Create videos directory
        self.video_dir = os.path.join(self.experiment_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(self.experiment_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=4)
        
        # Set up logging
        self.logger = self._setup_logging()
        self.logger.info(f"Starting experiment, saving to {self.experiment_dir}")
        self.logger.info(f"Configuration: {config}")
        
        # Set seed for reproducibility
        seed = config.get("seed", 42)
        set_seed(seed)
        
        # Set device
        device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)
        self.logger.info(f"Using device: {self.device}")
        
        # Create environments
        self.env, self.eval_env = self._create_environments()
        
        # Create agent
        self.agent = agent_class(self.env, config, self.device)
        
        # Training parameters
        self.num_episodes = config.get("num_episodes", 1000)
        self.max_steps_per_episode = config.get("max_steps_per_episode", 10000)
        self.evaluation_freq = config.get("evaluation_freq", 50)
        self.num_evaluation_episodes = config.get("num_evaluation_episodes", 10)
        self.log_freq = config.get("log_freq", 10)
        self.save_freq = config.get("save_freq", 100)
        self.eval_frequency_frames = config.get("eval_frequency_frames", 1000000)
        self.save_frequency_frames = config.get("save_frequency_frames", 2000000)
        self.max_frames = config.get("max_frames", 50000000)

        self.total_frames = 0
        self.eval_frames = []
        
        # Initialize basic tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_episodes = []
        self.training_losses = []
        self.q_values = []
        
        # Initialize advanced metrics tracking
        self.mean_td_errors = []
        self.max_td_errors = []
        self.weight_variance = []
        self.memory_sizes = []
        self.beta_values = []
        self.network_distances = []
        self.epsilon_values = []
    
        # Initialize advanced metrics tracking with flexible dictionary approach
        self.metrics = {
            "mean_td_errors": [],
            "max_td_errors": [],
            "weight_variance": [],
            "memory_sizes": [],
            "beta_values": [],
            "network_distances": [],
            # Standard metrics (may be unused by some agents)
            "epsilon_values": [],
            # NoisyNet specific metrics
            "weight_sigma_values": [],
            "bias_sigma_values": [],
            "noise_magnitude_values": []
        }
        
        # Initialize plotter
        self.plot_dir = os.path.join(self.experiment_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        self.plotter = RLPlotter(self.plot_dir, config.get("env_name", "Unknown"))
        
        self.timer = Timer()
        
    def _create_experiment_dir(self, base_dir, experiment_name):
        """Create directory for experiment results."""
        os.makedirs(base_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
        
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir
        
    def _setup_logging(self):
        """Set up logging to file and console."""
        log_file = os.path.join(self.experiment_dir, "train.log")
        
        # Create logger
        logger = logging.getLogger(f"trainer_{id(self)}")  # Unique logger per instance
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatters and handlers
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Prevent propagation to the root logger
        logger.propagate = False
        
        return logger
        
    def _create_environments(self):
        """Create training and evaluation environments."""
        env_name = self.config.get("env_name", "ALE/Pong-v5")
        frame_skip = self.config.get("frame_skip", 4)
        frame_stack = self.config.get("frame_stack", 4)
        episode_life = self.config.get("episode_life", True)
        clip_rewards = self.config.get("clip_rewards", True)
        
        # Training environment
        env = AtariWrapper(
            env_name=env_name,
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            episode_life=episode_life,
            clip_rewards=clip_rewards
        )
        
        # Evaluation environment
        eval_env = AtariWrapper(
            env_name=env_name,
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            episode_life=False,  # Don't terminate on life loss for evaluation
            clip_rewards=False   # Don't clip rewards for evaluation
        )
        
        return env, eval_env
        
    def preprocess_observation(self, observation):
        """Preprocess observation for agent input."""
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        
        # Add batch dimension if not present
        if len(observation.shape) == 3:  # [C, H, W]
            observation = observation.reshape(1, *observation.shape)
        
        # Convert to torch tensor and move to device
        tensor = torch.FloatTensor(observation).to(self.device)
        return tensor
        
    def evaluate_agent(self, record=False, video_path=None):
        """Evaluate agent performance."""
        rewards = []
        steps_list = []
        q_values_list = []
        
        # Determine which environment to use
        if record and video_path:
            # Create video wrapper
            video_dir = os.path.dirname(video_path)
            os.makedirs(video_dir, exist_ok=True)
            
            eval_env = AtariWrapper(
                env_name=self.eval_env.env_name,
                frame_skip=self.eval_env.frame_skip,
                frame_stack=self.eval_env.frame_stack,
                episode_life=False,
                clip_rewards=False,
                render_mode="rgb_array"
            )
            
            # Add recording wrapper
            video_name = os.path.basename(video_path)
            eval_env = RecordVideo(
                eval_env.env,
                video_folder=video_dir,
                name_prefix=video_name,
                episode_trigger=lambda x: True
            )
        else:
            eval_env = self.eval_env
        
        for episode in range(self.num_evaluation_episodes):
            state, _ = eval_env.reset()
            state = self.preprocess_observation(state)
            done = False
            episode_reward = 0
            episode_steps = 0
            episode_q_values = []
            
            while not done:
                # Get Q-values if the agent has a policy_net attribute
                if hasattr(self.agent, 'policy_net'):
                    with torch.no_grad():
                        q_values = self.agent.policy_net(state)
                        q_value = q_values.max().item()
                        episode_q_values.append(q_value)
                
                # Select action
                action = self.agent.select_action(state, training=False)
                
                # Execute action
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1
                
                if not done:
                    next_state = self.preprocess_observation(next_state)
                    state = next_state
            
            rewards.append(episode_reward)
            steps_list.append(episode_steps)
            if episode_q_values:
                q_values_list.append(np.mean(episode_q_values))
        
        if record and video_path:
            eval_env.close()
        
        # Handle case where agent doesn't expose q-values
        if not q_values_list:
            q_values_list = [0.0] * len(rewards)
        
        return np.mean(rewards), np.std(rewards), np.mean(steps_list), np.mean(q_values_list)
        
    def create_plots(self):
        """Create and save training plots using RLPlotter."""
        buffer_capacity = getattr(self.agent, 'buffer_size', 100000)
        
        # Combine all metrics into a single dictionary
        training_data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_losses": self.training_losses,
            "eval_episodes": self.eval_episodes,
            "eval_rewards": self.eval_rewards,
            "q_values": self.q_values,
        }
        # Add all tracked metrics
        for metric_name, metric_values in self.metrics.items():
            if metric_values:  # Only add non-empty metrics
                training_data[metric_name.replace("_values", "")] = metric_values
        
        # Use the plotter to create all plots
        self.plotter.create_all_plots(training_data, buffer_capacity)
        
    def record_video(self):
        """Record a video of the agent playing."""
        try:
            # Create an environment with render mode
            env_name = self.config.get("env_name", "ALE/Pong-v5")
            
            # Define preprocessing function for frames
            def preprocess_frame(frame):
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = np.mean(frame, axis=2, keepdims=False)
                
                frame = Image.fromarray(frame)
                frame = frame.convert('L')
                frame = frame.resize((84, 84))
                frame = np.array(frame) / 255.0
                
                return frame
            
            # Create environment with rendering
            env = AtariWrapper(
                env_name=env_name,
                frame_skip=self.env.frame_skip,
                frame_stack=self.env.frame_stack,
                episode_life=False,
                clip_rewards=False,
                render_mode="rgb_array"
            )
            
            # Wrap for recording
            env = RecordVideo(
                env.env,
                video_folder=self.video_dir, 
                name_prefix="demo",
                episode_trigger=lambda _: True
            )
            
            # Play one episode
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            step = 0
            
            # Convert observation to agent's format
            state = self.preprocess_observation(obs)
            
            while not done and step < 10000:
                # Select action
                with torch.no_grad():
                    action = self.agent.select_action(state, training=False)
                
                # Take step in environment
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Process next observation
                if not done:
                    state = self.preprocess_observation(next_obs)
                
                step += 1
                
                # Clean up CUDA memory if needed
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            self.logger.info(f"Recorded episode with reward {episode_reward} in {step} steps")
            env.close()
            
        except Exception as e:
            self.logger.error(f"Error recording video: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                env.close()
            except:
                pass
        
        return self.video_dir
        
    def train(self):
        """Train the agent and evaluate periodically."""
        self.timer.start()
        episode = 0
        next_eval_frame = self.eval_frequency_frames
        next_save_frame = self.save_frequency_frames
        
        while self.total_frames < self.max_frames:
            episode += 1
            state, _ = self.env.reset()
            state = self.preprocess_observation(state)
            
            done = False
            episode_reward = 0
            episode_loss = []
            steps = 0
            
            while not done and steps < self.max_steps_per_episode:
                # For agents that implement store_transition
                if hasattr(self.agent, "store_transition"):
                    # Select action
                    action = self.agent.select_action(state)
                    
                    # Execute action
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    # Process next state
                    if not done:
                        next_state = self.preprocess_observation(next_state)
                    else:
                        next_state = None
                    
                    # Store transition using agent's method
                    self.agent.store_transition(state, action, reward, next_state, done)
                    
                    # Move to the next state
                    state = next_state if next_state is not None else state
                    
                else:
                    # For agents with traditional memory.push approach
                    # Select action
                    action = self.agent.select_action(state)
                    
                    # Execute action
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    # Process next state
                    if not done:
                        next_state = self.preprocess_observation(next_state)
                    else:
                        next_state = None
                    
                    # Store transition in replay buffer
                    self.agent.memory.push(state, action, reward, next_state, done)
                    
                    # Move to the next state
                    state = next_state if next_state is not None else state
                
                # Get the most recent training metrics from the agent
                latest_metrics = self.agent.train()

                # Collect any metrics returned by the agent - more flexible approach
                if latest_metrics:
                    for metric_name, metric_value in latest_metrics.items():
                        # Check if the metric is something we want to track
                        if metric_value is not None:
                            # Convert to our naming convention for storage
                            storage_name = f"{metric_name}_values" if not metric_name.endswith("_values") else metric_name
                            
                            # Create storage if it doesn't exist yet
                            if storage_name not in self.metrics:
                                self.metrics[storage_name] = []
                                
                            # Store the metric
                            self.metrics[storage_name].append(metric_value)
                            
                            # Special case for backward compatibility
                            if metric_name == "epsilon":
                                self.epsilon_values.append(metric_value)  # Legacy support
                            elif metric_name == "mean_td_error":
                                self.mean_td_errors.append(metric_value)  # Legacy support
                            elif metric_name == "max_td_error":
                                self.max_td_errors.append(metric_value)  # Legacy support
                            elif metric_name == "weight_variance":
                                self.weight_variance.append(metric_value)  # Legacy support
                            elif metric_name == "memory_size":
                                self.memory_sizes.append(metric_value)  # Legacy support
                            elif metric_name == "beta":
                                self.beta_values.append(metric_value)  # Legacy support
                            elif metric_name == "network_distance":
                                self.network_distances.append(metric_value)  # Legacy support
                # Update Counters
                episode_reward += reward
                steps += 1
                self.total_frames += 1

                # Check for evaluation based on frame count
                if self.total_frames >= next_eval_frame:
                    self.logger.info(f"Evaluating after {self.total_frames} frames (episode {episode})...")
                    
                    # Record video periodically
                    video_record_freq = self.config.get("video_record_frequency", 5)
                    record_video = self.total_frames == self.eval_frequency_frames or next_eval_frame % (self.eval_frequency_frames * video_record_freq) == 0
                    
                    # Evaluate
                    mean_reward, std_reward, mean_steps, mean_q = self.evaluate_agent(
                        record=record_video,
                        video_path=os.path.join(self.video_dir, f"eval_frame_{self.total_frames}")
                    )
                    
                    # Store metrics
                    self.eval_rewards.append(mean_reward)
                    self.eval_frames.append(self.total_frames)  # Use frames instead of episodes
                    self.q_values.append(mean_q)
                    
                    # Log evaluation results
                    self.logger.info(
                        f"Evaluation at {self.total_frames} frames: "
                        f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f} | "
                        f"Mean Q-value: {mean_q:.4f}"
                    )
                    
                    # Create plots during training
                    self.create_plots()
                    
                    # Set next evaluation point
                    next_eval_frame += self.eval_frequency_frames
                
                # Check for saving based on frame count
                if self.total_frames >= next_save_frame:
                    save_path = os.path.join(self.experiment_dir, f"agent_frame_{self.total_frames}.pt")
                    self.agent.save(save_path)
                    self.logger.info(f"Saved agent to {save_path} after {self.total_frames} frames")
                    next_save_frame += self.save_frequency_frames

            
            # Track episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            if hasattr(self.agent, "update_epsilon"):
                _ = self.agent.update_epsilon()
            
            # Track loss and TD error metrics
            if episode_loss:
                avg_loss = np.mean(episode_loss)
                self.training_losses.append(avg_loss)
            else:
                avg_loss = float('nan')
                
            # Log progress
            if (episode + 1) % self.log_freq == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_freq:])
                avg_length = np.mean(self.episode_lengths[-self.log_freq:])
                
                log_msg = [
                    f"Episode {episode+1}/{self.num_episodes}",
                    f"Reward: {episode_reward:.1f}",
                    f"Avg Reward: {avg_reward:.1f}",
                    f"Steps: {steps}",
                ]

                if episode_loss:
                    log_msg.append(f"Avg Loss: {avg_loss:.4f}")
                # Add any advanced metrics if available
                if hasattr(self.agent, "epsilon"):
                    log_msg.append(f"Epsilon: {self.agent.epsilon:.4f}")
                elif latest_metrics and "weight_sigma" in latest_metrics:
                    log_msg.append(f"Noise σ: {latest_metrics['weight_sigma']:.4f}")
                
                log_msg.append(f"Elapsed: {self.timer.elapsed:.1f}s")
                self.logger.info(" | ".join(log_msg))
            
            # Evaluate agent and collect advanced metrics
            if (episode + 1) % self.evaluation_freq == 0:
                # Record video every 5 evaluations
                video_record_freq = self.config.get("video_record_frequency", 5)
                record_video = episode == 0 or (episode + 1) % (self.evaluation_freq * video_record_freq) == 0
                video_path = None
                
                if record_video:
                    video_path = os.path.join(self.video_dir, f"eval_episode_{episode+1}")
                    
                mean_reward, std_reward, mean_steps, mean_q = self.evaluate_agent(
                    record=record_video,
                    video_path=video_path
                )
                
                # Store basic evaluation metrics
                self.eval_rewards.append(mean_reward)
                self.eval_episodes.append(episode + 1)
                self.q_values.append(mean_q)
                
                # Get the most recent training metrics from the agent
                latest_metrics = self.agent.train()
                
                # Collect advanced metrics if available
                if latest_metrics:
                    if "mean_td_error" in latest_metrics:
                        self.mean_td_errors.append(latest_metrics["mean_td_error"])
                    if "max_td_error" in latest_metrics:
                        self.max_td_errors.append(latest_metrics["max_td_error"])
                    if "weight_variance" in latest_metrics:
                        self.weight_variance.append(latest_metrics["weight_variance"])
                    if "memory_size" in latest_metrics:
                        self.memory_sizes.append(latest_metrics["memory_size"])
                    if "beta" in latest_metrics and latest_metrics["beta"] is not None:
                        self.beta_values.append(latest_metrics["beta"])
                    if "network_distance" in latest_metrics and latest_metrics["network_distance"] is not None:
                        self.network_distances.append(latest_metrics["network_distance"])
                    if "epsilon" in latest_metrics:
                        self.epsilon_values.append(latest_metrics["epsilon"])
                
                # Log evaluation results
                self.logger.info(
                    f"Evaluation after episode {episode+1}: "
                    f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f} | "
                    f"Mean steps: {mean_steps:.1f} | Mean Q-value: {mean_q:.4f}"
                )
                
                # Create plots during training
                self.create_plots()
            
            # Save agent
            if (episode + 1) % self.save_freq == 0:
                save_path = os.path.join(self.experiment_dir, f"agent_episode_{episode+1}.pt")
                self.agent.save(save_path)
                self.logger.info(f"Saved agent to {save_path}")
            
            # Update timer
            self.timer.update()
        
        # Final evaluation
        self.logger.info("Performing final evaluation...")
        final_video_path = os.path.join(self.video_dir, "final_evaluation")
        mean_reward, std_reward, mean_steps, mean_q = self.evaluate_agent(
            record=True, 
            video_path=final_video_path
        )
        
        self.logger.info(
            f"Final evaluation: Mean reward: {mean_reward:.1f} ± {std_reward:.1f} | "
            f"Mean steps: {mean_steps:.1f} | Mean Q-value: {mean_q:.4f}"
        )
        
        # Save final agent
        save_path = os.path.join(self.experiment_dir, "agent_final.pt")
        self.agent.save(save_path)
        self.logger.info(f"Saved final agent to {save_path}")
        
        # Save all training data including advanced metrics
        training_data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_losses": self.training_losses,
            "eval_rewards": self.eval_rewards,
            "eval_episodes": self.eval_episodes,
            "q_values": self.q_values,
        }

        # Add all tracked metrics
        for metric_name, metric_values in self.metrics.items():
            if metric_values:  # Only add non-empty metrics
                # Store with consistent naming
                clean_name = metric_name.replace("_values", "")
                training_data[clean_name] = metric_values
        
        with open(os.path.join(self.experiment_dir, "training_data.json"), 'w') as f:
            json.dump(training_data, f)
        
        # Generate comprehensive reports
        self.create_plots()
        
        # Close environments
        self.env.close()
        self.eval_env.close()
        
        # Record a demo video with the final agent
        try:
            self.logger.info("Recording demo video with final agent...")
            demo_video_path = self.record_video()
            self.logger.info(f"Demo video saved to {demo_video_path}")
        except Exception as e:
            self.logger.error(f"Error recording demo video: {e}")
            self.logger.info("Continuing without demo video")
        
        self.logger.info(f"Training completed! Results saved to {self.experiment_dir}")
        
        return {
            "agent": self.agent,
            "training_data": training_data,
            "experiment_dir": self.experiment_dir
        }