import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from core.agent import Agent
from core.networks import DQNNetwork
from core.buffer import ReplayBuffer

class DQNAgent(Agent):
    """Deep Q-Network agent"""
    
    def __init__(self, env, config, device="cpu"):
        """Initialize the DQN agent
        
        Args:
            env: Environment to interact with
            config: Agent configuration
            device: Device to run the agent on
        """
        super(DQNAgent, self).__init__(env, device)
        
        # Extract configuration
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 1e-4)
        self.batch_size = config.get("batch_size", 32)
        self.target_update_freq = config.get("target_update_freq", 10000)
        self.double_dqn = config.get("double_dqn", True)
        
        # Epsilon greedy exploration
        self.epsilon_start = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.1)
        self.epsilon_decay = config.get("epsilon_decay", 1000)
        self.epsilon = self.epsilon_start
        
        # Get state and action dimensions
        state_shape = env.get_state_shape()
        num_actions = env.action_space.n
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_shape, num_actions, device).to(device)
        self.target_net = DQNNetwork(state_shape, num_actions, device).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Initialize replay buffer
        buffer_size = config.get("buffer_size", 100000)
        self.memory = ReplayBuffer(buffer_size, device)
        
        # Initialize step counter
        self.steps_done = 0
        self.episodes_done = 0
    
    def update_epsilon(self):
        """Update epsilon after completing an episode"""
        self.episodes_done += 1
        
        # Linear decay based on episodes
        decay_rate = min(1.0, self.episodes_done / self.epsilon_decay)
        self.epsilon = max(
            self.epsilon_end, 
            self.epsilon_start + decay_rate * (self.epsilon_end - self.epsilon_start)
        )
        
        return self.epsilon
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether the agent is training (use epsilon-greedy) or evaluating (greedy)
            
        Returns:
            Selected action
        """
        # Decay epsilon
        if training:
            self.steps_done += 1
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Random action
            return random.randrange(self.env.action_space.n)
        else:
            # Greedy action
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
    
    def train(self, experiences=None):
        """Train the agent on a batch of experiences
        
        Args:
            experiences: Optional batch of experiences. If None, sample from replay buffer.
            
        Returns:
            Dictionary of training metrics
        """
        # Skip if we don't have enough samples in the buffer
        if len(self.memory) < self.batch_size:
            return {"loss": None}
        
        # Sample from replay buffer
        if experiences is None:
            batch = self.memory.sample(self.batch_size)
            state, action, reward, non_final_next_states, non_final_mask, done = batch
        else:
            state, action, reward, non_final_next_states, non_final_mask, done = experiences
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        
        # Compute V(s_{t+1}) for all next states using target network
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        
        if self.double_dqn:
            # Double DQN: use policy net to select action and target net to evaluate it
            with torch.no_grad():
                # Get actions from policy network
                next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                
                # Evaluate actions using target network
                next_state_values[non_final_mask] = self.target_net(non_final_next_states) \
                                                   .gather(1, next_actions).squeeze(1)
        else:
            # Standard DQN: use target net for both action selection and evaluation
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = reward + (self.gamma * next_state_values * ~done)
        
        # Compute loss
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        # Update target network if needed
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()
        
        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "q_values": state_action_values.mean().item()
        }
    
    def update_target_network(self):
        """Update the target network with the policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        """Save the agent to the specified path
        
        Args:
            path: Path to save the agent to
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load the agent from the specified path
        
        Args:
            path: Path to load the agent from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file found at {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']