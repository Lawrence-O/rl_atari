#!/usr/bin/env python3
"""
Deep Q-Network (DQN) implementation for Atari games.
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, AtariPreprocessing, FrameStackObservation
import yaml
from PIL import Image

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.environment import AtariWrapper
from algorithms.dqn.dqn_agent import DQNAgent
from core.utils import set_seed


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari games")
    parser.add_argument("--config", type=str, default="experiments/configs/dqn_pong.yaml", 
                      help="Path to the configuration file")
    parser.add_argument("--experiment_name", type=str, default=None,
                      help="Name of the experiment (default: dqn_<env_name>_<timestamp>)")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(experiment_dir):
    """Set up logging to file and console."""
    log_file = os.path.join(experiment_dir, "train.log")
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatters and handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def create_experiment_dir(base_dir, experiment_name=None):
    """Create directory for experiment results."""
    os.makedirs(base_dir, exist_ok=True)
    
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(base_dir, f"dqn_{timestamp}")
    else:
        experiment_dir = os.path.join(base_dir, experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def preprocess_observation(observation, device=None):
    """Preprocess observation for DQN input."""
    if not isinstance(observation, np.ndarray):
        observation = np.array(observation)
    
    # Add batch dimension if not present
    if len(observation.shape) == 3:  # [C, H, W]
        observation = observation.reshape(1, *observation.shape)
    
    # Convert to torch tensor
    tensor = torch.FloatTensor(observation)
    
    # Move to device if specified
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def evaluate_agent(agent, env, num_episodes=10, record=False, video_path=None):
    """Evaluate agent performance."""
    rewards = []
    steps_list = []
    q_values_list = []
    device = agent.device
    
    # Determine which environment to use
    if record and video_path:
        # Create video directory if it doesn't exist
        video_dir = os.path.dirname(video_path)
        os.makedirs(video_dir, exist_ok=True)
        
        # Create a new environment with render_mode set to rgb_array
        base_env = gym.make(env.env_name, render_mode="rgb_array", frameskip=1)
        
        # Apply the same preprocessing
        eval_env = AtariWrapper(
            env_name=env.env_name,
            frame_skip=env.frame_skip,
            frame_stack=env.frame_stack,
            episode_life=False,  # Don't terminate on life loss for evaluation
            clip_rewards=False,  # Don't clip rewards for evaluation
            render_mode="rgb_array"  # Important: set render mode
        )
        
        # Add recording wrapper to the base env (not wrapped env)
        video_name = os.path.basename(video_path)
        eval_env = RecordVideo(
            eval_env.env,
            video_folder=video_dir,
            name_prefix=video_name,
            episode_trigger=lambda x: True  # Record all episodes
        )
    else:
        eval_env = env
    
    for episode in range(num_episodes):
        state, _ = eval_env.reset()
        state = preprocess_observation(state, device)
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_q_values = []
        
        while not done:
            # Get Q-values for the current state
            with torch.no_grad():
                q_values = agent.policy_net(state)
                q_value = q_values.max().item()
                episode_q_values.append(q_value)
            
            # Select action
            action = agent.select_action(state, training=False)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            
            if not done:
                next_state = preprocess_observation(next_state, device)
                state = next_state
        
        rewards.append(episode_reward)
        steps_list.append(episode_steps)
        q_values_list.append(np.mean(episode_q_values))
    
    if record and video_path:
        eval_env.close()
    
    return np.mean(rewards), np.std(rewards), np.mean(steps_list), np.mean(q_values_list)


def create_plots(data, env_name, output_dir):
    """Create and save training plots."""
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Set plot style
    sns.set(style="darkgrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    
    # Plot rewards
    plt.figure()
    plt.plot(data["episode_rewards"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Training rewards - {env_name}")
    plt.savefig(os.path.join(plot_dir, "rewards.png"))
    
    # Plot smoothed rewards
    plt.figure()
    window = min(50, len(data["episode_rewards"]) // 5)
    smoothed = np.convolve(data["episode_rewards"], np.ones(window)/window, mode='valid')
    plt.plot(data["episode_rewards"], alpha=0.3, color='blue')
    plt.plot(range(window-1, window-1+len(smoothed)), smoothed, color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Smoothed training rewards - {env_name}")
    plt.savefig(os.path.join(plot_dir, "rewards_smoothed.png"))
    
    # Plot losses if available
    if "training_losses" in data and data["training_losses"]:
        plt.figure()
        plt.plot(data["training_losses"])
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.title(f"Training loss - {env_name}")
        plt.savefig(os.path.join(plot_dir, "loss.png"))
    
    # Plot evaluation rewards if available
    if "eval_rewards" in data and "eval_episodes" in data:
        plt.figure()
        plt.plot(data["eval_episodes"], data["eval_rewards"], 'o-')
        plt.xlabel("Episode")
        plt.ylabel("Evaluation Reward")
        plt.title(f"Evaluation rewards - {env_name}")
        plt.savefig(os.path.join(plot_dir, "eval_rewards.png"))
    
    # Plot Q-values if available
    if "q_values" in data and data["q_values"]:
        plt.figure()
        plt.plot(data["eval_episodes"], data["q_values"], 'o-')
        plt.xlabel("Episode")
        plt.ylabel("Average Q-Value")
        plt.title(f"Q-value evolution - {env_name}")
        plt.savefig(os.path.join(plot_dir, "q_values.png"))
    
    plt.close('all')


def record_agent_video(agent, env_name, video_dir, device):
    """Record a video of the agent playing."""
    # Create video folder if it doesn't exist
    os.makedirs(video_dir, exist_ok=True)
    
    # Save original device
    original_device = next(agent.policy_net.parameters()).device
    
    try:

        # Create an environment with render mode
        env = gym.make(env_name, render_mode="rgb_array")
        
        # Wrap for recording
        env = RecordVideo(env, video_folder=video_dir, name_prefix="demo", episode_trigger=lambda _: True)
        
        # We'll stack frames manually for more control
        frame_stack = []
        frame_stack_size = 4  # Match the agent's expected input
        
        # Play one episode
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        # Pre-fill frame stack with first observation
        frame = preprocess_frame(obs)
        for _ in range(frame_stack_size):
            frame_stack.append(frame.copy())
        
        while not done and step < 10000:  # Limit to 10000 steps
            # Prepare state for agent (ensuring 4D tensor with batch dimension)
            stacked_frames = np.stack(frame_stack, axis=0)  # Stack along channel dimension
            state = torch.FloatTensor(stacked_frames).unsqueeze(0).to(device)  # [1, 4, H, W]
            
            # Select action with no gradient tracking
            with torch.no_grad():
                action = agent.select_action(state, training=False)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Process new frame and update frame stack
            if not done:
                frame = preprocess_frame(next_obs)
                frame_stack.pop(0)  # Remove oldest frame
                frame_stack.append(frame)  # Add new frame
            
            step += 1
            
            # Clean up CUDA memory if needed
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        print(f"Recorded episode with reward {episode_reward} in {step} steps")
        env.close()
        print(f"Video saved to {video_dir}")
        
    except Exception as e:
        print(f"Error recording video: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Make sure environment is closed
        try:
            env.close()
        except:
            pass
        
        # Move model back to original device if needed
        if device != original_device:
            agent.policy_net.to(original_device)
            agent.target_net.to(original_device)
    
    return video_dir


def preprocess_frame(frame):
    """Preprocess a single frame for DQN."""
    # If RGB, convert to grayscale
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = np.mean(frame, axis=2, keepdims=False)
    
    # Resize to 84x84
    frame = Image.fromarray(frame)
    frame = frame.convert('L')
    frame = frame.resize((84, 84))
    frame = np.array(frame) / 255.0
    
    return frame


def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up experiment directory
    experiment_name = args.experiment_name or f"dqn_{config['env_name'].split('/')[-1].lower()}_run1"
    experiment_dir = create_experiment_dir("experiments/results", experiment_name)
    
    # Create videos directory
    video_dir = os.path.join(experiment_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(experiment_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Set up logging
    logger = setup_logging(experiment_dir)
    logger.info(f"Starting experiment, saving to {experiment_dir}")
    logger.info(f"Configuration: {config}")
    
    # Set seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Set device
    device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    logger.info(f"Using device: {device}")
    
    # Create environment
    env_name = config.get("env_name", "ALE/Pong-v5")
    frame_skip = config.get("frame_skip", 4)
    frame_stack = config.get("frame_stack", 4)
    episode_life = config.get("episode_life", True)
    clip_rewards = config.get("clip_rewards", True)
    
    env = AtariWrapper(
        env_name=env_name,
        frame_skip=frame_skip,
        frame_stack=frame_stack,
        episode_life=episode_life,
        clip_rewards=clip_rewards
    )
    
    # Create evaluation environment (without episode termination on life loss)
    eval_env = AtariWrapper(
        env_name=env_name,
        frame_skip=frame_skip,
        frame_stack=frame_stack,
        episode_life=False,  # Don't terminate on life loss for evaluation
        clip_rewards=False   # Don't clip rewards for evaluation
    )
    
    # Create agent
    agent = DQNAgent(env, config, device)
    
    # Training parameters
    num_episodes = config.get("num_episodes", 1000)
    max_steps_per_episode = config.get("max_steps_per_episode", 10000)
    evaluation_freq = config.get("evaluation_freq", 50)
    num_evaluation_episodes = config.get("num_evaluation_episodes", 10)
    log_freq = config.get("log_freq", 10)
    save_freq = config.get("save_freq", 100)
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    eval_episodes = []
    training_losses = []
    q_values = []  # Track Q-values from evaluation
    
    timer = Timer()
    timer.start()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_observation(state, device)
        
        done = False
        episode_reward = 0
        episode_loss = []
        steps = 0
        
        while not done and steps < max_steps_per_episode:
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if not done:
                next_state = preprocess_observation(next_state, device)
            else:
                next_state = None
            
            # Store transition in replay buffer
            agent.memory.push(state, action, reward, next_state, done)
            
            # Move to the next state
            state = next_state if next_state is not None else state
            
            # Train agent
            metrics = agent.train()
            if metrics["loss"] is not None:
                episode_loss.append(metrics["loss"])
            
            episode_reward += reward
            steps += 1
        
        # Track episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        _ = agent.update_epsilon()
        
        if episode_loss:
            avg_loss = np.mean(episode_loss)
            training_losses.append(avg_loss)
        else:
            avg_loss = float('nan')
        
        # Log progress
        if (episode + 1) % log_freq == 0:
            avg_reward = np.mean(episode_rewards[-log_freq:])
            avg_length = np.mean(episode_lengths[-log_freq:])
            epsilon = agent.epsilon
            
            logger.info(
                f"Episode {episode+1}/{num_episodes} | "
                f"Reward: {episode_reward:.1f} | "
                f"Avg Reward: {avg_reward:.1f} | "
                f"Steps: {steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"Epsilon: {epsilon:.4f} | "
                f"Elapsed: {timer.elapsed:.1f}s"
            )
        
        # Evaluate agent
        if (episode + 1) % evaluation_freq == 0:
            # Only record video every 5 evaluations (to save space)
            record_video = (episode + 1) % (evaluation_freq * 5) == 0
            video_path = None
            
            if record_video:
                video_path = os.path.join(video_dir, f"eval_episode_{episode+1}")
                
            mean_reward, std_reward, mean_steps, mean_q = evaluate_agent(
                agent, eval_env, 
                num_episodes=num_evaluation_episodes,
                record=record_video,
                video_path=video_path
            )
            
            eval_rewards.append(mean_reward)
            eval_episodes.append(episode + 1)
            q_values.append(mean_q)  # Track average Q-values
            
            logger.info(
                f"Evaluation after episode {episode+1}: "
                f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f} | "
                f"Mean steps: {mean_steps:.1f} | Mean Q-value: {mean_q:.4f}"
            )
            
            # Create intermediate plots during training
            training_data = {
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "training_losses": training_losses,
                "eval_rewards": eval_rewards,
                "eval_episodes": eval_episodes,
                "q_values": q_values
            }
            
            # Create basic plots during training
            create_plots(
                training_data, 
                env_name.split('/')[-1],  # Just use the game name
                experiment_dir
            )
        
        # Save agent
        if (episode + 1) % save_freq == 0:
            save_path = os.path.join(experiment_dir, f"agent_episode_{episode+1}.pt")
            agent.save(save_path)
            logger.info(f"Saved agent to {save_path}")
        
        # Update timer
        timer.update()
    
    # Final evaluation
    logger.info("Performing final evaluation...")
    final_video_path = os.path.join(video_dir, "final_evaluation")
    mean_reward, std_reward, mean_steps, mean_q = evaluate_agent(
        agent, eval_env, 
        num_episodes=num_evaluation_episodes,
        record=True, 
        video_path=final_video_path
    )
    
    logger.info(
        f"Final evaluation: Mean reward: {mean_reward:.1f} ± {std_reward:.1f} | "
        f"Mean steps: {mean_steps:.1f} | Mean Q-value: {mean_q:.4f}"
    )
    
    # Save final agent
    save_path = os.path.join(experiment_dir, "agent_final.pt")
    agent.save(save_path)
    logger.info(f"Saved final agent to {save_path}")
    
    # Save all training data
    training_data = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "training_losses": training_losses,
        "eval_rewards": eval_rewards,
        "eval_episodes": eval_episodes,
        "q_values": q_values
    }
    
    with open(os.path.join(experiment_dir, "training_data.json"), 'w') as f:
        json.dump(training_data, f)
    
    # Generate comprehensive reports
    create_plots(training_data, env_name.split('/')[-1], experiment_dir)
    
    # Close environments
    env.close()
    eval_env.close()
    
    # Record a demo video with the final agent
    try:
        logger.info("Recording demo video with final agent...")
        import cv2  # Import here for the preprocess_frame function
        demo_video_path = record_agent_video(agent, env_name, video_dir, device)
        logger.info(f"Demo video saved to {demo_video_path}")
    except Exception as e:
        logger.error(f"Error recording demo video: {e}")
        logger.info("Continuing without demo video")
    
    logger.info(f"Training completed! Results saved to {experiment_dir}")


if __name__ == "__main__":
    main()