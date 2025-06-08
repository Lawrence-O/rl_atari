import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation
import torch
from ale_py import ALEInterface

class AtariWrapper:
    """Atari environment wrapper with standard preprocessing"""
    def __init__(self, env_name, frame_skip=4, frame_stack=4, episode_life=True, clip_rewards=True, render_mode=None):
        self.env_name = env_name
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.episode_life = episode_life
        self.clip_rewards = clip_rewards
        self.env = self._make_env()
        
        # Environment properties
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def _make_env(self):
        """Create and wrap the Atari environment"""
        ale = ALEInterface()
        gym.register_envs(ale)
        
        # Create environment with frameskip=1 to avoid double frame-skipping
        env = gym.make(self.env_name, render_mode=self.render_mode, frameskip=1)
        
        # Apply standard Atari preprocessing
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=self.frame_skip,
            screen_size=84,
            terminal_on_life_loss=self.episode_life,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=True
        )
        
        # Stack frames to observe motion using Gymnasium's wrapper
        if self.frame_stack > 1:
            env = FrameStackObservation(env, self.frame_stack)
        
        return env
    
    def reset(self):
        """Reset the environment"""
        return self.env.reset()
    
    def step(self, action):
        """Take a step in the environment"""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        # Clip rewards for stability if requested
        if self.clip_rewards:
            reward = np.sign(reward)
            
        return next_state, reward, terminated, truncated, info
    
    def close(self):
        """Close the environment"""
        self.env.close()
        
    def get_state_shape(self):
        """Get the shape of the state tensor"""
        return self.observation_space.shape

def preprocess_observation(observation, device=None):
    """Convert observation to PyTorch tensor on the appropriate device
    
    Args:
        observation: Observation to preprocess
        device: Device to move the tensor to (None for CPU)
        
    Returns:
        Preprocessed observation as a PyTorch tensor
    """
    # Convert to numpy array if not already
    if not isinstance(observation, np.ndarray):
        observation = np.array(observation)
    
    # Print shape for debugging
    # print(f"Original observation shape: {observation.shape}")
    
    # Reshape based on the observation's shape
    if len(observation.shape) == 1:  # Vector observation (e.g., CartPole)
        # Add batch dimension for vector observations
        observation = observation.reshape(1, -1)
    elif len(observation.shape) == 5:  # [batch, frames, height, width, channel]
        # Convert to [batch, channels, height, width] for CNN
        observation = np.transpose(observation, (0, 4, 1, 2, 3))  
        observation = observation.reshape(observation.shape[0], -1, observation.shape[3], observation.shape[4])
    elif len(observation.shape) == 4:  # [frames, height, width, channel] or [batch, channel, height, width]
        if observation.shape[3] == 1:  # Likely [frames, height, width, channel]
            # Convert to [batch, channels, height, width] for CNN
            observation = np.transpose(observation, (3, 0, 1, 2))  
            observation = observation.reshape(1, -1, observation.shape[2], observation.shape[3])
        # Otherwise already in correct format [batch, channel, height, width]
    elif len(observation.shape) == 3:  # [channel, height, width] or [height, width, channel]
        if observation.shape[0] != 1 and observation.shape[0] != 3 and observation.shape[0] != 4:
            # Likely [height, width, channel]
            observation = np.transpose(observation, (2, 0, 1))  
        # Add batch dimension
        observation = observation[np.newaxis, :]
    
    # Convert to tensor
    tensor = torch.FloatTensor(observation)
    
    # Move to specified device if provided
    if device is not None:
        tensor = tensor.to(device)
    
    # Print shape for debugging  
    # print(f"Processed tensor shape: {tensor.shape}")
        
    return tensor

class ClassicControlWrapper:
    """Wrapper for Classic Control environments like CartPole"""
    
    def __init__(self, env_name, render_mode=None):
        """Initialize the environment
        
        Args:
            env_name: Name of the environment
            render_mode: Render mode for visualization
        """
        self.env_name = env_name
        self.render_mode = render_mode
        self.env = self._make_env()
        
        # Track environment properties
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.frame_skip = 1  # No frame skipping for classic control
        self.frame_stack = 1  # No frame stacking for classic control
    
    def _make_env(self):
        """Create the environment"""
        env = gym.make(self.env_name, render_mode=self.render_mode)
        return env
    
    def reset(self):
        """Reset the environment"""
        return self.env.reset()
    
    def step(self, action):
        """Take a step in the environment"""
        return self.env.step(action)
    
    def close(self):
        """Close the environment"""
        return self.env.close()
    
    def get_state_shape(self):
        """Get the shape of the state"""
        if isinstance(self.observation_space, gym.spaces.Box):
            return self.observation_space.shape[0]  # Return the flat dimension for vector inputs
        return self.observation_space.shape