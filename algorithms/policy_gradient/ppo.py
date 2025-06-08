import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.agent import Agent
from core.networks import PolicyNetwork
from torch.distributions.categorical import Categorical

class PPOAgent(Agent):
    """Proximal Policy Optimization (PPO) agent"""

    def __init__(self, env, config, device="cpu"):
        """Initialize the PPO agent
        
        Args:
            env: Environment to interact with
            config: Agent configuration
            device: Device to run the agent on
        """
        super(PPOAgent, self).__init__(env, device)

        # Extract configuration
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 3e-4)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.batch_size = config.get("batch_size", 64)
        self.update_epochs = config.get("update_epochs", 10)

        # Get state and action dimensions
        state_shape = env.get_state_shape()
        num_actions = env.action_space.n

        # Initialize policy and value networks
        self.policy_net = PolicyNetwork(state_shape, num_actions, device).to(device)
        self.value_net = PolicyNetwork(state_shape, 1, device).to(device)  
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

