import torch
import numpy as np
from abc import ABC, abstractmethod

class Agent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, env, device):
        """Initialize agent
        
        Args:
            env: Environment the agent interacts with
            device: Device to run the agent on
        """
        self.env = env
        self.device = device
        
    @abstractmethod
    def select_action(self, state, training=True):
        """Select an action given the current state
        
        Args:
            state: Current state
            training: Whether the agent is training or evaluating
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def train(self, experience):
        """Train the agent on a batch of experiences
        
        Args:
            experience: Batch of experiences
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """Save the agent to the specified path
        
        Args:
            path: Path to save the agent to
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """Load the agent from the specified path
        
        Args:
            path: Path to load the agent from
        """
        pass