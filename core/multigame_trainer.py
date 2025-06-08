import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
import json
import gymnasium as gym
import logging
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

from core.cuda_manager import CudaStreamManager
from core.multigame_network import MultiGameNetwork
from core.environment import AtariWrapper, preprocess_observation
from core.buffer import PrioritizedReplayBuffer
from core.utils import set_seed, create_experiment_dir
from core.plotting import RLPlotter

class MultiGameTrainer:
    """Trainer for multi-game representation learning with CUDA acceleration"""
    
    def __init__(self, config):
        """Initialize the multi-game trainer with CUDA optimizations
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.max_frames = config.get("max_frames", 50000000)
        
        # Setup logging and experiment directory
        exp_name = config.get("exp_name", f"multigame_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir = create_experiment_dir("experiments/results", exp_name)
        self.logger = self._setup_logging()
        
        # Save configuration
        with open(os.path.join(self.experiment_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=4)
            
        # Set random seed
        seed = config.get("seed", 42)
        set_seed(seed)
        
        # Create video directory
        self.video_dir = os.path.join(self.experiment_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        
        # Setup games and environments
        self.games = config.get("games", ["Pong", "Breakout", "SpaceInvaders", "Phoenix"])
        self.logger.info(f"Training on {len(self.games)} games: {', '.join(self.games)}")
        
        # Create environments
        self._setup_environments()
        
        # Create network
        self._setup_network()
        
        # Set up CUDA optimizations
        self._setup_cuda_optimizations()
        
        # Create replay buffers
        self._setup_replay_buffers()
        
        # Set up training parameters
        self._setup_training_params()
        
        # Set up tracking metrics
        self._setup_metrics()
        
        self.logger.info("Multi-game trainer initialized")
    
    def _setup_environments(self):
        """Set up training and evaluation environments for each game"""
        self.envs = {}
        self.eval_envs = {}
        self.game_action_spaces = {}
        
        for game in self.games:
            env_name = f"ALE/{game}-v5"
            # Create training environment
            self.envs[game] = AtariWrapper(
                env_name=env_name,
                frame_skip=self.config.get("frame_skip", 4),
                frame_stack=self.config.get("frame_stack", 4),
                episode_life=self.config.get("episode_life", True),
                clip_rewards=self.config.get("clip_rewards", True)
            )
            
            # Create evaluation environment
            self.eval_envs[game] = AtariWrapper(
                env_name=env_name,
                frame_skip=self.config.get("frame_skip", 4),
                frame_stack=self.config.get("frame_stack", 4),
                episode_life=False,  # Don't terminate on life loss for evaluation
                clip_rewards=False   # Don't clip rewards for evaluation
            )
            
            self.game_action_spaces[game] = self.envs[game].action_space.n
            
        self.input_shape = (self.config.get("frame_stack", 4), 84, 84)
    
    def _setup_network(self):
        """Initialize the network architecture"""
        # Create shared network with game-specific heads
        self.policy_net = MultiGameNetwork(self.input_shape, self.game_action_spaces, 
                                          self.config.get("feature_dim", 512))
        self.target_net = MultiGameNetwork(self.input_shape, self.game_action_spaces,
                                          self.config.get("feature_dim", 512))
        
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def _setup_cuda_optimizations(self):
        """Set up CUDA optimizations if available"""
        if self.device.type == "cuda":
            self.logger.info("Setting up CUDA optimizations")
            self.stream_manager = CudaStreamManager(self.games)
            
            # Capture backbone graph if requested
            if self.config.get("use_cuda_graph", True):
                self.logger.info("Capturing CUDA graph for backbone forward pass")
                try:
                    self.stream_manager.capture_backbone_graph(
                        self.policy_net.shared_backbone, 
                        self.input_shape,
                        self.config.get("batch_size", 32)
                    )
                    self.logger.info("Successfully captured CUDA graph")
                except Exception as e:
                    self.logger.warning(f"Failed to capture CUDA graph: {e}")
    
    def _setup_replay_buffers(self):
        """Initialize replay buffers for each game"""
        self.replay_start_size = self.config.get("replay_start_size", 20000)
        buffer_size = self.config.get("buffer_size", 100000)
        alpha = self.config.get("alpha", 0.6)
        beta = self.config.get("beta", 0.4)
        self.beta_annealing = self.config.get("beta_annealing", 10000000)
        
        self.memory = {}
        for game in self.games:
            self.memory[game] = PrioritizedReplayBuffer(
                buffer_size, 
                alpha=alpha, 
                beta=beta,
                device=self.device
            )
    
    def _setup_training_params(self):
        """Set up training parameters"""
        # Core training parameters
        self.batch_size = self.config.get("batch_size", 32)
        self.gamma = self.config.get("gamma", 0.99)
        self.lr = self.config.get("learning_rate", 0.0000625)
        self.n_steps = self.config.get("n_steps", 3)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.target_update_freq = self.config.get("target_update_freq", 10000)
        
        # Evaluation parameters
        self.eval_frequency = self.config.get("eval_frequency_frames", 250000)
        self.num_eval_episodes = self.config.get("num_evaluation_episodes", 10)
        self.save_frequency = self.config.get("save_frequency_frames", 1000000)
    
    def _setup_metrics(self):
        """Initialize metrics tracking"""
        # Frame tracking
        self.total_frames = 0
        self.frames_per_game = {game: 0 for game in self.games}
        self.updates = 0
        self.steps_done = {game: 0 for game in self.games}
        self.current_states = {}
        
        # Performance tracking
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_log_frames = 0
        
        # Training metrics
        self.metrics = {
            "rewards": defaultdict(list),
            "eval_rewards": defaultdict(list),
            "q_values": defaultdict(list),
            "loss": [],
            "frames": [],
            "fps": []
        }
        
        # Initialize plotters
        self.plot_dir = os.path.join(self.experiment_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        self.plotters = {
            game: RLPlotter(os.path.join(self.plot_dir, game), game) 
            for game in self.games
        }
    
    def _setup_logging(self):
        """Set up logging to file and console"""
        log_file = os.path.join(self.experiment_dir, "train.log")
        
        # Create logger
        logger = logging.getLogger(f"multigame_trainer_{id(self)}")
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
    
    def select_action(self, state, game, training=True):
        """Select an action given the current state for a specific game
        
        Args:
            state: Current state tensor
            game: Game name
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        # Reset noise before selecting action (per Rainbow paper)
        if hasattr(self.policy_net.game_heads[game], 'reset_noise'):
            self.policy_net.game_heads[game].reset_noise()
            
        if training:
            self.steps_done[game] += 1
        
        with torch.no_grad():
            # Get Q-values
            q_values = self.policy_net(state, game)
            # Select action with highest Q-value
            action = q_values.max(1)[1].item()
            
        return action
    
    def _optimize_model(self):
        """Train on batches from all games using CUDA acceleration"""
        # Prepare batches from each game
        batches = {}
        for game in self.games:
            # Skip games with insufficient data
            if len(self.memory[game]) < self.batch_size:
                continue
                
            batch = self.memory[game].sample(self.batch_size)
            if batch is None:
                continue
                
            state, action, reward, non_final_next_states, non_final_mask, done, weights, indices = batch
            batches[game] = {
                'states': state,
                'actions': action, 
                'rewards': reward,
                'next_states': non_final_next_states,
                'dones': done,
                'non_final_mask': non_final_mask,
                'weights': weights,
                'indices': indices
            }
        
        # Skip if no valid batches
        if not batches:
            return 0.0
            
        # Reset noise before forward pass
        for game in batches.keys():
            if hasattr(self.policy_net.game_heads[game], 'reset_noise'):
                self.policy_net.game_heads[game].reset_noise()
                self.target_net.game_heads[game].reset_noise()
        
        # Process batches and compute losses
        losses = self._compute_losses(batches)
        
        # Combined loss across all games
        total_loss = sum(losses) / len(losses)
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        # Update target network periodically
        self.updates += 1
        if self.updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.logger.info(f"Updated target network at {self.total_frames:,} frames")
        
        return total_loss.item()
    
    def _compute_losses(self, batches):
        """Compute losses for each game batch
        
        Args:
            batches: Dictionary of game batches
            
        Returns:
            List of loss tensors for each game
        """
        losses = []
        
        # Process games in parallel using CUDA streams if available
        if hasattr(self, 'stream_manager') and self.device.type == "cuda":
            # First pass for Q-values: Process state batches with policy network
            state_batches = {game: batch['states'] for game, batch in batches.items()}
            q_values = self.stream_manager.process_batch_with_streams(self.policy_net, state_batches)
            
            # Get max action indices for next states from policy network (double DQN)
            next_state_batches = {game: batch['next_states'] for game, batch in batches.items() 
                                if batch['non_final_mask'].any()}
            
            with torch.no_grad():
                # Get Q-values for next states from policy net
                next_q_values = self.stream_manager.process_batch_with_streams(
                    self.policy_net, next_state_batches) if next_state_batches else {}
                
                # Get max actions from policy net
                next_actions = {game: q.max(1)[1].unsqueeze(1) for game, q in next_q_values.items()}
                
                # Get Q-values for those actions from target net
                target_q_values = self.stream_manager.process_batch_with_streams(
                    self.target_net, next_state_batches) if next_state_batches else {}
        else:
            # Sequential processing without CUDA streams
            q_values = {}
            next_q_values = {}
            next_actions = {}
            target_q_values = {}
            
            for game, batch in batches.items():
                # Get current Q-values
                q_values[game] = self.policy_net(batch['states'], game)
                
                # Double DQN: Get actions from policy net
                with torch.no_grad():
                    if batch['non_final_mask'].any():
                        next_q = self.policy_net(batch['next_states'], game)
                        next_actions[game] = next_q.max(1)[1].unsqueeze(1)
                        
                        # Get Q-values for those actions from target net
                        target_q = self.target_net(batch['next_states'], game)
                        target_q_values[game] = target_q
        
        # Calculate loss for each game
        for game, batch in batches.items():
            # Get Q-values for taken actions
            state_action_values = q_values[game].gather(1, batch['actions'].unsqueeze(1)).squeeze(1)
            
            # Calculate target values
            with torch.no_grad():
                # Initialize next state values
                next_state_values = torch.zeros(self.batch_size, device=self.device)
                
                # Only compute for non-terminal states
                if batch['non_final_mask'].any() and game in next_actions and game in target_q_values:
                    next_values = target_q_values[game].gather(1, next_actions[game])
                    next_state_values[batch['non_final_mask']] = next_values.squeeze(1)
                
                # Compute expected Q values with discount
                expected_q = batch['rewards'] + (self.gamma ** self.n_steps) * next_state_values * ~batch['dones']
            
            # Compute loss with importance sampling weights
            td_errors = state_action_values - expected_q
            loss = (td_errors.pow(2) * batch['weights']).mean()
            losses.append(loss)
            
            # Update priorities
            priorities = td_errors.abs().detach().cpu() + 1e-6
            self.memory[game].update_priorities(batch['indices'], priorities)
        
        return losses
    
    def evaluate(self, game, num_episodes=10, record=False):
        """Evaluate the agent on a specific game
        
        Args:
            game: Game to evaluate on
            num_episodes: Number of episodes to evaluate
            record: Whether to record video
            
        Returns:
            Mean reward, mean Q-values, and total evaluation frames
        """
        rewards = []
        q_values_list = []
        total_eval_frames = 0
        
        if record:
            # For recording, we'll use standard gymnasium wrappers
            from gymnasium.wrappers import AtariPreprocessing, FrameStack
            
            # 1. Create base Atari environment with rendering enabled
            env_name = f"ALE/{game}-v5"
            base_env = gym.make(
                env_name,
                render_mode="rgb_array",  # Required for recording
                frameskip=1               # Will be handled by AtariPreprocessing
            )
            
            # 2. Create timestamped video directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.video_dir, f"{game}_{self.total_frames}_{timestamp}")
            os.makedirs(video_path, exist_ok=True)
            
            # 3. First apply standard Atari preprocessing
            processed_env = AtariPreprocessing(
                base_env,
                noop_max=30,
                frame_skip=self.config.get("frame_skip", 4),
                screen_size=84,
                terminal_on_life_loss=False,  # No life loss termination for evaluation
                grayscale_obs=True,
                grayscale_newaxis=False,
                scale_obs=True
            )
            
            # 4. Apply frame stacking
            stacked_env = FrameStack(processed_env, self.config.get("frame_stack", 4))
            
            # 5. Apply recording wrapper last (so it records the processed frames)
            env = gym.wrappers.RecordVideo(
                stacked_env,
                video_folder=video_path,
                name_prefix=game,
                episode_trigger=lambda _: True
            )
            
            self.logger.info(f"Recording evaluation video for {game} to {video_path}")
        else:
            # Use standard evaluation environment
            env = self.eval_envs[game]
        
        try:
            for episode in range(num_episodes):
                state, _ = env.reset()
                state = preprocess_observation(state, self.device)
                done = False
                episode_reward = 0
                episode_q_values = []
                episode_frames = 0
                
                while not done:
                    # Get Q-values and select action
                    with torch.no_grad():
                        q_values = self.policy_net(state, game)
                        episode_q_values.append(q_values.max().item())
                        action = q_values.max(1)[1].item()
                    
                    # Execute action
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    episode_frames += 1
                    
                    if not done:
                        state = preprocess_observation(next_state, self.device)
                
                rewards.append(episode_reward)
                total_eval_frames += episode_frames
                if episode_q_values:
                    q_values_list.append(np.mean(episode_q_values))
        
        finally:
            # Clean up recording environment
            if record:
                env.close()
                self.logger.info(f"Completed recording for {game}")
        
        return np.mean(rewards), np.mean(q_values_list) if q_values_list else 0.0, total_eval_frames
    
    def train(self):
        """Train the shared network across multiple games"""
        self.logger.info(f"Starting training on {len(self.games)} games for {self.max_frames:,} frames")
        self.logger.info(f"Replay buffer starts training at {self.replay_start_size:,} frames per game")
        
        # Initialize states for each game
        for game in self.games:
            state, _ = self.envs[game].reset()
            self.current_states[game] = preprocess_observation(state, self.device)
        
        start_time = time.time()
        next_eval_frame = self.eval_frequency
        next_save_frame = self.save_frequency
        current_rewards = {game: 0 for game in self.games}
        reward_window = {game: [] for game in self.games}
        
        # Main training loop
        while self.total_frames < self.max_frames:
            # Collect experience from each game
            for game in self.games:
                state = self.current_states[game]
                
                # Select action
                action = self.select_action(state, game, training=True)
                
                # Take step in environment
                next_state, reward, terminated, truncated, _ = self.envs[game].step(action)
                done = terminated or truncated
                
                # Process next state
                if done:
                    next_state_tensor = None
                else:
                    next_state_tensor = preprocess_observation(next_state, self.device)
                
                # Store transition
                self.memory[game].push(state, action, reward, next_state_tensor, done)
                
                # Update state
                self.current_states[game] = next_state_tensor if next_state_tensor is not None else state
                
                # Track rewards and frames per game
                current_rewards[game] += reward
                self.frames_per_game[game] += 1
                
                # Reset environment if done
                if done:
                    # Track rewards in window
                    reward_window[game].append(current_rewards[game])
                    if len(reward_window[game]) > 100:
                        reward_window[game].pop(0)
                    
                    # Store metrics every 1000 frames
                    if self.frames_per_game[game] % 1000 < 4:  # Close enough to 1000 threshold
                        self.metrics["rewards"][game].append(
                            (self.frames_per_game[game], np.mean(reward_window[game]))
                        )
                    
                    # Log completion
                    self.logger.info(
                        f"Game: {game} | "
                        f"Frames: {self.frames_per_game[game]:,} | "
                        f"Reward: {current_rewards[game]:.1f} | "
                        f"Avg100: {np.mean(reward_window[game]):.1f}"
                    )
                    
                    # Reset current reward and environment
                    current_rewards[game] = 0
                    state, _ = self.envs[game].reset()
                    self.current_states[game] = preprocess_observation(state, self.device)
            
            # Update total frames
            self.total_frames += len(self.games)
            
            # Anneal beta for PER
            beta_progress = min(1.0, self.total_frames / self.beta_annealing)
            for game in self.games:
                if hasattr(self.memory[game], 'beta'):
                    self.memory[game].beta = 0.4 + beta_progress * 0.6  # Anneal from 0.4 to 1.0
            
            # Skip training until all buffers are filled
            buffers_ready = all(len(mem) >= self.replay_start_size for mem in self.memory.values())
            if not buffers_ready:
                if self.total_frames % 10000 == 0:
                    buffer_sizes = {g: f"{len(self.memory[g]):,}" for g in self.games}
                    self.logger.info(f"Filling buffers: {buffer_sizes} / {self.replay_start_size:,}")
                continue
            
            # Training step
            loss = self._optimize_model()
            
            # Track metrics
            if self.total_frames % 1000 == 0:
                current_time = time.time()
                elapsed = current_time - self.start_time
                recent_elapsed = current_time - self.last_log_time
                recent_frames = self.total_frames - self.last_log_frames
                
                # Calculate FPS
                overall_fps = self.total_frames / elapsed if elapsed > 0 else 0
                recent_fps = recent_frames / recent_elapsed if recent_elapsed > 0 else 0
                
                self.metrics["loss"].append((self.total_frames, loss))
                self.metrics["frames"].append(self.total_frames)
                self.metrics["fps"].append((self.total_frames, recent_fps))
                
                # Update for next calculation
                self.last_log_time = current_time
                self.last_log_frames = self.total_frames
            
            # Log progress
            if self.total_frames % 10000 == 0:
                elapsed = time.time() - start_time
                overall_fps = self.total_frames / elapsed if elapsed > 0 else 0
                
                # Frame distribution across games
                frame_pcts = {game: f"{frames/self.total_frames*100:.1f}%" 
                              for game, frames in self.frames_per_game.items()}
                
                self.logger.info(
                    f"Frames: {self.total_frames:,}/{self.max_frames:,} ({self.total_frames/self.max_frames*100:.1f}%) | "
                    f"Loss: {loss:.5f} | FPS: {overall_fps:.1f}"
                )
                self.logger.info(f"Frame distribution: {frame_pcts}")
            
            # Periodic evaluation
            if self.total_frames >= next_eval_frame:
                self.logger.info(f"Evaluating at {self.total_frames:,} frames")
                
                total_eval_frames = 0
                for game in self.games:
                    record = (self.total_frames == self.eval_frequency or 
                              self.total_frames % (self.eval_frequency * 5) == 0)
                    
                    mean_reward, mean_q, game_eval_frames = self.evaluate(
                        game, self.num_eval_episodes, record=record
                    )
                    total_eval_frames += game_eval_frames
                    
                    self.metrics["eval_rewards"][game].append((self.total_frames, mean_reward))
                    self.metrics["q_values"][game].append((self.total_frames, mean_q))
                    
                    self.logger.info(
                        f"Game: {game} | "
                        f"Mean reward: {mean_reward:.1f} | "
                        f"Mean Q-value: {mean_q:.4f} | "
                        f"Eval frames: {game_eval_frames:,}"
                    )
                
                # Create plots
                self.create_plots()
                
                self.logger.info(f"Evaluation complete - {total_eval_frames:,} total eval frames")
                next_eval_frame = self.total_frames + self.eval_frequency
            
            # Save checkpoint
            if self.total_frames >= next_save_frame:
                save_path = os.path.join(self.experiment_dir, f"model_{self.total_frames}.pt")
                self.save(save_path)
                self.logger.info(f"Saved checkpoint to {save_path}")
                
                # Create plots
                self.create_plots()
                
                next_save_frame = self.total_frames + self.save_frequency
        
        # Final evaluation and saving
        self.logger.info("Training complete! Performing final evaluation...")
        final_metrics = {}
        total_final_eval_frames = 0
        
        for game in self.games:
            mean_reward, mean_q, game_eval_frames = self.evaluate(
                game, self.num_eval_episodes, record=True
            )
            final_metrics[game] = {
                "reward": mean_reward,
                "q_value": mean_q,
                "total_frames": self.frames_per_game[game],
                "eval_frames": game_eval_frames
            }
            total_final_eval_frames += game_eval_frames
            
            self.logger.info(
                f"Final evaluation - Game: {game} | "
                f"Mean reward: {mean_reward:.1f} | "
                f"Mean Q-value: {mean_q:.4f} | "
                f"Total frames: {self.frames_per_game[game]:,}"
            )
        
        # Save final model
        final_path = os.path.join(self.experiment_dir, "model_final.pt")
        self.save(final_path)
        self.logger.info(f"Saved final model to {final_path}")
        
        # Create final plots
        self.create_plots()
        
        # Save training data
        self.save_training_data()
        
        # Log performance summary
        elapsed = time.time() - start_time
        self.logger.info(f"===== TRAINING SUMMARY =====")
        self.logger.info(f"Total frames: {self.total_frames:,}")
        self.logger.info(f"Training time: {elapsed:.1f}s ({self.total_frames/elapsed:.1f} FPS)")
        self.logger.info(f"Frame distribution:")
        
        for game, frames in self.frames_per_game.items():
            self.logger.info(f"  {game}: {frames:,} ({frames/self.total_frames*100:.1f}%)")
        
        return {
            "metrics": self.metrics,
            "final_metrics": final_metrics,
            "training_time": elapsed,
            "total_frames": self.total_frames,
            "frames_per_game": self.frames_per_game
        }
    
    def create_plots(self):
        """Create frame-based plots for each game and combined metrics"""
        # First create individual game plots
        for game in self.games:
            training_data = {
                "frames_vs_rewards": self.metrics["rewards"][game],
                "frames_vs_eval_rewards": self.metrics["eval_rewards"][game],
                "frames_vs_q_values": self.metrics["q_values"][game],
                "frames": self.metrics["frames"],
                "loss": self.metrics["loss"],
                "fps": self.metrics["fps"]
            }
            
            self.plotters[game].create_frame_plots(training_data, self.memory[game].buffer.maxlen)
        
        # Create combined plots
        self._create_combined_plots()
    
    def _create_combined_plots(self):
        """Create combined frame-based plots comparing games"""
        # Rewards comparison
        plt.figure(figsize=(12, 8))
        for game in self.games:
            if self.metrics["eval_rewards"][game]:
                frames, rewards = zip(*self.metrics["eval_rewards"][game])
                plt.plot(frames, rewards, label=game)
        
        plt.xlabel("Frames")
        plt.ylabel("Evaluation Reward")
        plt.title("Evaluation Rewards Across Games")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.plot_dir, "combined_rewards.png"))
        plt.close()
        
        # Q-values comparison
        plt.figure(figsize=(12, 8))
        for game in self.games:
            if self.metrics["q_values"][game]:
                frames, q_values = zip(*self.metrics["q_values"][game])
                plt.plot(frames, q_values, label=game)
        
        plt.xlabel("Frames")
        plt.ylabel("Q-Value")
        plt.title("Q-Values Across Games")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.plot_dir, "combined_q_values.png"))
        plt.close()
        
        # FPS over time
        plt.figure(figsize=(12, 8))
        if self.metrics["fps"]:
            frames, fps_values = zip(*self.metrics["fps"])
            plt.plot(frames, fps_values)
            plt.axhline(y=np.mean(fps_values), color='r', linestyle='--', 
                       label=f"Avg: {np.mean(fps_values):.1f} FPS")
        
        plt.xlabel("Frames")
        plt.ylabel("Frames Per Second")
        plt.title("Training Speed")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.plot_dir, "fps.png"))
        plt.close()
        
        # Frame distribution pie chart
        plt.figure(figsize=(10, 10))
        plt.pie(
            [self.frames_per_game[game] for game in self.games],
            labels=[f"{game}\n{self.frames_per_game[game]:,} frames" for game in self.games],
            autopct='%1.1f%%'
        )
        plt.title("Frame Distribution Across Games")
        plt.savefig(os.path.join(self.plot_dir, "frame_distribution.png"))
        plt.close()
    
    def save_training_data(self):
        """Save frame-based training data to JSON for later analysis"""
        data_path = os.path.join(self.experiment_dir, "training_data.json")
        
        # Convert defaultdict to regular dict for JSON serialization
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, defaultdict):
                serializable_metrics[key] = {k: v for k, v in value.items()}
            else:
                serializable_metrics[key] = value
        
        # Add frame statistics
        serializable_metrics["frames_per_game"] = self.frames_per_game
        serializable_metrics["total_frames"] = self.total_frames
        
        with open(data_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        
        self.logger.info(f"Saved training data to {data_path}")
    
    def save(self, path):
        """Save model and optimizer state with frame counts"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_frames': self.total_frames,
            'frames_per_game': self.frames_per_game,
            'updates': self.updates,
            'config': self.config
        }, path)
    
    def load(self, path):
        """Load model and optimizer state with frame counts"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.total_frames = checkpoint.get('total_frames', 0)
        self.frames_per_game = checkpoint.get('frames_per_game', {game: 0 for game in self.games})
        self.updates = checkpoint.get('updates', 0)
        
        self.logger.info(f"Loaded model from {path} at {self.total_frames:,} frames")
        
        return checkpoint