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
from core.plotting import RLPlotter


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
    """Generic trainer for reinforcement learning agents using frame-based evaluation."""
    
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
        self.max_steps_per_episode = config.get("max_steps_per_episode", 10000)
        self.num_evaluation_episodes = config.get("num_evaluation_episodes", 10)
        self.log_freq = config.get("log_freq", 10)
        
        # Frame-based parameters
        self.eval_frequency_frames = config.get("eval_frequency_frames", 1000000)
        self.save_frequency_frames = config.get("save_frequency_frames", 2000000)
        self.max_frames = config.get("max_frames", 50000000)

        # Frame counter
        self.total_frames = 0
        
        # Initialize basic tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_frames = []
        self.training_losses = []
        self.q_values = []
        
        # Initialize metrics tracking with flexible dictionary approach
        self.metrics = {
            "mean_td_errors": [],
            "max_td_errors": [],
            "weight_variance": [],
            "memory_sizes": [],
            "beta_values": [],
            "network_distances": [],
            "epsilon_values": [],
            "weight_sigmas": [],
            "bias_sigmas": [],
            "noise_magnitudes": []
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
            "eval_frames": self.eval_frames,  # Use frames as x-axis
            "eval_rewards": self.eval_rewards,
            "q_values": self.q_values,
        }
        
        # Add all tracked metrics
        for metric_name, metric_values in self.metrics.items():
            if metric_values:  # Only add non-empty metrics
                # Store with consistent naming
                clean_name = metric_name.replace("_values", "")
                training_data[clean_name] = metric_values
        
        # Use the plotter to create all plots
        self.plotter.create_all_plots(training_data, buffer_capacity)
        
    def record_video(self):
        """Record a video of the agent playing."""
        try:
            # Create an environment with render mode
            env_name = self.config.get("env_name", "ALE/Pong-v5")
            
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
        """Train the agent using frame-based evaluation."""
        self.timer.start()
        self.logger.info(f"Starting training for {self.max_frames:,} frames")
        
        episode = 0
        next_eval_frame = self.eval_frequency_frames
        next_save_frame = self.save_frequency_frames
        
        # Main training loop - continue until max frames is reached
        while self.total_frames < self.max_frames:
            episode += 1
            state, _ = self.env.reset()
            state = self.preprocess_observation(state)
            
            done = False
            episode_reward = 0
            episode_loss = []
            steps = 0
            
            # Episode loop
            while not done and steps < self.max_steps_per_episode:
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
                
                # Store transition - support both API styles
                if hasattr(self.agent, "store_transition"):
                    self.agent.store_transition(state, action, reward, next_state, done)
                else:
                    self.agent.memory.push(state, action, reward, next_state, done)
                
                # Move to the next state
                state = next_state if next_state is not None else state
                
                # Train the agent and get metrics
                latest_metrics = self.agent.train()

                # Track buffer size and target if available
                if latest_metrics and "buffer_size" in latest_metrics and "buffer_target" in latest_metrics:
                    if self.total_frames % 5000 == 0 or latest_metrics["buffer_size"] == latest_metrics["buffer_target"]:
                        self.logger.info(
                            f"Collecting experiences: {latest_metrics['buffer_size']:,}/{latest_metrics['buffer_target']:,} "
                            f"frames ({latest_metrics['buffer_size']/latest_metrics['buffer_target']*100:.1f}%) - "
                            f"Learning will start at {latest_metrics['buffer_target']:,} frames"
                        )

                # Track loss if available
                if latest_metrics and 'loss' in latest_metrics and latest_metrics['loss'] is not None:
                    episode_loss.append(latest_metrics['loss'])
                    self.training_losses.append(latest_metrics['loss'])

                # Collect all metrics
                if latest_metrics:
                    for metric_name, metric_value in latest_metrics.items():
                        if metric_value is not None:
                            # Standardize naming convention
                            storage_name = metric_name + "s" if not metric_name.endswith("s") else metric_name

                            # Ensure storage exists
                            if storage_name not in self.metrics:
                                self.metrics[storage_name] = []
                                
                            # Store the metric
                            self.metrics[storage_name].append(metric_value)
                
                # Update counters
                episode_reward += reward
                steps += 1
                self.total_frames += 1

                # Check if we've reached max frames
                if self.total_frames >= self.max_frames:
                    self.logger.info(f"Reached frame limit of {self.max_frames:,}, stopping training")
                    break

                # Check for frame-based evaluation
                if self.total_frames >= next_eval_frame:
                    eval_start_time = time.time()
                    self.logger.info(f"Starting evaluation at {self.total_frames:,} frames (episode {episode})")
                    
                    # Log training progress details
                    buffer_size = len(self.agent.memory) if hasattr(self.agent, "memory") else "unknown"
                    agent_updates = getattr(self.agent, "updates", self.total_frames)
                    
                    # Determine if we should record video
                    video_record_freq = self.config.get("video_record_frequency", 5)
                    record_video = (
                        self.total_frames == self.eval_frequency_frames or 
                        (self.total_frames % (self.eval_frequency_frames * video_record_freq)) == 0
                    )
                    
                    # Evaluate agent
                    mean_reward, std_reward, mean_steps, mean_q = self.evaluate_agent(
                        record=record_video,
                        video_path=os.path.join(self.video_dir, f"eval_frame_{self.total_frames}")
                    )
                    
                    eval_time = time.time() - eval_start_time
                    
                    # Store evaluation metrics
                    self.eval_rewards.append(mean_reward)
                    self.eval_frames.append(self.total_frames)
                    self.q_values.append(mean_q)
                    
                    # Enhanced evaluation log
                    self.logger.info(
                        f"Evaluation at {self.total_frames:,} frames complete: \n"
                        f"  Reward: {mean_reward:.1f} ± {std_reward:.1f} \n"
                        f"  Steps: {mean_steps:.1f} \n"
                        f"  Q-value: {mean_q:.4f} \n"
                        f"  Buffer size: {buffer_size:,} \n"
                        f"  Agent updates: {agent_updates:,} \n"
                        f"  Eval time: {eval_time:.2f}s \n"
                        f"  Training progress: {self.total_frames/self.max_frames*100:.1f}% complete \n"
                        f"  Elapsed time: {self.timer.elapsed:.1f}s"
                    )
                    
                    # Add time projection to completion
                    if self.total_frames > 0:
                        frames_per_second = self.total_frames / self.timer.elapsed if self.timer.elapsed > 0 else 0
                        remaining_frames = self.max_frames - self.total_frames
                        est_remaining_time = remaining_frames / frames_per_second if frames_per_second > 0 else float('inf')
                        
                        # Format time estimate
                        if est_remaining_time < 60*60:  # Less than an hour
                            time_str = f"{est_remaining_time/60:.1f} minutes"
                        elif est_remaining_time < 24*60*60:  # Less than a day
                            time_str = f"{est_remaining_time/3600:.1f} hours"
                        else:
                            time_str = f"{est_remaining_time/(24*3600):.1f} days"
                            
                        self.logger.info(f"Estimated time to completion: {time_str} (at {frames_per_second:.1f} frames/sec)")
                    
                    # Create plots
                    self.create_plots()
                    
                    # Set next evaluation point
                    next_eval_frame += self.eval_frequency_frames
                
                # Check for frame-based model saving
                if self.total_frames >= next_save_frame:
                    save_path = os.path.join(self.experiment_dir, f"agent_frame_{self.total_frames}.pt")
                    self.agent.save(save_path)
                    self.logger.info(f"Saved agent to {save_path} at {self.total_frames:,} frames")
                    next_save_frame += self.save_frequency_frames

            # Track episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
                
            # Log episode progress
            if (episode + 1) % self.log_freq == 0:
                # Calculate average metrics for logging
                avg_reward = np.mean(self.episode_rewards[-self.log_freq:])
                avg_length = np.mean(self.episode_lengths[-self.log_freq:])
                
                # Format loss string if available
                if episode_loss:
                    avg_loss = np.mean(episode_loss)
                    loss_str = f"Loss: {avg_loss:.4f} | "
                else:
                    loss_str = ""
                
                # Format TD error string if available
                td_error_str = ""
                if "mean_td_errors" in self.metrics and self.metrics["mean_td_errors"]:
                    td_error_str = f"TD Error: {self.metrics['mean_td_errors'][-1]:.4f} | "
                
                # Log comprehensive progress
                self.logger.info(
                    f"Episode {episode} | "
                    f"Frames: {self.total_frames:,}/{self.max_frames:,} ({self.total_frames/self.max_frames*100:.1f}%) | "
                    f"Reward: {episode_reward:.1f} | "
                    f"Avg Reward: {avg_reward:.1f} | "
                    f"{loss_str}"
                    f"{td_error_str}"
                    f"Steps: {steps} | "
                    f"Elapsed: {self.timer.elapsed:.1f}s"
                )
            
            # Update timer
            self.timer.update()
        
        # Final evaluation
        self.logger.info("Training complete! Performing final evaluation...")
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
        
        # Save training data
        training_data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_losses": self.training_losses,
            "eval_frames": self.eval_frames,
            "eval_rewards": self.eval_rewards,
            "q_values": self.q_values,
        }
        
        # Add all tracked metrics
        for metric_name, metric_values in self.metrics.items():
            if metric_values:  # Only add non-empty metrics
                clean_name = metric_name.replace("_values", "")
                training_data[clean_name] = metric_values
        
        with open(os.path.join(self.experiment_dir, "training_data.json"), 'w') as f:
            json.dump(training_data, f)
        
        # Create final plots
        self.create_plots()
        
        # Close environments
        self.env.close()
        self.eval_env.close()
        
        # Record a demo video
        try:
            self.logger.info("Recording demo video with final agent...")
            demo_video_path = self.record_video()
            self.logger.info(f"Demo video saved to {demo_video_path}")
        except Exception as e:
            self.logger.error(f"Error recording demo video: {e}")
            self.logger.info("Continuing without demo video")
        
        self.logger.info(f"Training completed! Total frames: {self.total_frames:,}")
        self.logger.info(f"Results saved to {self.experiment_dir}")
        
        return {
            "agent": self.agent,
            "training_data": training_data,
            "experiment_dir": self.experiment_dir
        }