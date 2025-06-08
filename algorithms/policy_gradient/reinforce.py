import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from core.agent import Agent
from core.networks import PolicyNetwork


class ReinforceAgent(Agent):
    """REINFORCE agent with baseline"""

    def __init__(self, env, config, device="cpu"):
        """Initialize the REINFORCE agent

        Args:
            env: Environment to interact with
            config: Agent configuration
            device: Device to run the agent on
        """
        super(ReinforceAgent, self).__init__(env, device)

        # Extract configuration
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 1e-3)
        self.batch_size = config.get("batch_size", 32)
        self.hidden_dim = config.get("hidden_dim", 128)
        self.base_line = config.get("base_line", True)

        # Get state and action dimensions
        state_shape = env.get_state_shape()
        num_actions = env.action_space.n

        # Initialize policy network
        self.policy_net = PolicyNetwork(state_shape, num_actions, self.hidden_dim, device).to(device)
        self.value_net = PolicyNetwork(state_shape, 1, self.hidden_dim, device).to(device)

        # Initialize optimizer
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)

        # Initialize step counter and device
        self.steps_done = 0
        self.device = device
        
        # Initialize episode buffer for collecting full episodes
        self.current_episode = []
        self.episodes_seen = 0
        
        # Metrics for tracking
        self.entropy_history = []
        self.policy_grad_norms = []
        self.value_grad_norms = []
    
    def select_action(self, state, training=True):
        """Select an action based on the current state using the policy network
        Args:
            state: Current state of the environment
            training: Whether in training mode (for exploration)
        Returns:
            action: Selected action
        """
        # Get action logits from the policy network
        action_logits = self.policy_net(state)
        # Convert to probabilities
        action_probs = F.softmax(action_logits, dim=-1)

        # if random.random() < 0.0005:  # Only print occasionally
        #     print(f"Action probs: {action_probs.detach().cpu().numpy()}")
        
        action = torch.multinomial(action_probs, 1).item()  
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in the current episode buffer
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Store transition in current episode
        self.current_episode.append((state, action, reward, next_state, done))
        
        # If episode is complete, train on it
        if done:
            self.episodes_seen += 1
            # Process the complete episode
            training_metrics = self.train(self.current_episode)
            # Clear episode buffer for next episode
            self.current_episode = []
            return training_metrics
        
        # Return None if episode is not complete
        return None
    
    def train(self, episode=None):
        """Train the agent using the REINFORCE algorithm
        Args:
            episode: Episode data containing states, actions, rewards, next_states, and dones
        Returns:
            Dict of training metrics
        """
        if episode is None or len(episode) == 0:
            return None
            
        # Unpack experience
        states, actions, rewards, next_states, dones = zip(*episode)

        # Convert to tensors
        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Compute returns
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * ~d
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Get baseline values if using baseline
        baseline = self.value_net(states).squeeze() if self.base_line else torch.zeros_like(returns).to(self.device)
        
        # Advantage estimates
        advantages = returns - baseline
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages
        
        # Track advantage statistics for plotting
        advantage_mean = advantages.mean().item()
        advantage_std = advantages.std().item()
        advantage_values = advantages.detach().cpu().numpy().tolist()

        # Compute Policy Loss
        action_logits = self.policy_net(states)
        probs = F.softmax(action_logits, dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze()  
        
        # Calculate entropy for monitoring exploration
        entropy = -(probs * F.log_softmax(action_logits, dim=-1)).sum(dim=-1).mean()
        self.entropy_history.append(entropy.item())

        policy_loss = (-torch.mean(log_probs * advantages.detach())) + 0.01 * entropy
   
        # Compute Value Loss
        value_loss = F.mse_loss(self.value_net(states).squeeze(), returns.detach())

        # Optimize policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # Get policy gradient norm for monitoring
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.policy_grad_norms.append(policy_grad_norm.item())
        self.policy_optimizer.step()
        
        # Optimize value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        # Get value gradient norm for monitoring
        value_grad_norm = torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 10)
        self.value_grad_norms.append(value_grad_norm.item())
        self.value_optimizer.step()

        self.steps_done += 1
        
        # Return training metrics
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "mean_return": returns.mean().item(),
            "mean_advantage": advantage_mean,
            "advantage_std": advantage_std,
            "advantage_values": advantage_values,
            "policy_grad_norm": policy_grad_norm.item(),
            "value_grad_norm": value_grad_norm.item(),
            "steps_done": self.steps_done,
            "episode_length": len(episode),
            "episodes_seen": self.episodes_seen,
            "returns": returns.detach().cpu().numpy().tolist()
        }
    
    def save(self, path):
        """Save the model to disk
        
        Args:
            path: Path to save the model to
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_seen': self.episodes_seen,
        }, path)
    
    def load(self, path):
        """Load the model from disk
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.steps_done = checkpoint['steps_done']
        if 'episodes_seen' in checkpoint:
            self.episodes_seen = checkpoint['episodes_seen']