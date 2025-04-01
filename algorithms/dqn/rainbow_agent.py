from collections import deque
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from core.agent import Agent
from core.networks import DuelingNoisyNetwork, NoisyLinear
from core.buffer import PrioritizedReplayBuffer


class RainbowAgent(Agent):
    def __init__(self, env, config, device="cpu"):
        """Initialize Rainbow agent
        
        Args:
            env: Environment the agent interacts with
            config: Agent configuration
            device: Device to run the agent on
            
        """
        super(RainbowAgent, self).__init__(env, device)
        
        # Extract configuration
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 1e-4)
        self.batch_size = config.get("batch_size", 32)
        self.target_update_freq = config.get("target_update_freq", 10000)

        # Prioritized replay buffer
        self.buffer_size = config.get("buffer_size", 100000)
        self.alpha = config.get("alpha", 0.6) # Priority exponent
        self.beta = config.get("beta", 0.4) # Importance sampling exponent
        self.td_error_epsilon = config.get("td_error_epsilon", 1e-6)
        self.beta_annealing = config.get("beta_annealing", 100000) # Steps to anneal beta to 1.0

        # Get state and action dimensions
        state_shape = env.get_state_shape()
        num_actions = env.action_space.n

        # N step learning
        self.n_steps = config.get("n_steps", 3)
        self.n_step_buffer = deque(maxlen=self.n_steps)

        # Initialize networks
        self.policy_net = DuelingNoisyNetwork(state_shape, num_actions, device).to(device)
        self.target_net = DuelingNoisyNetwork(state_shape, num_actions, device).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Initialize replay buffer
        self.memory = PrioritizedReplayBuffer(self.buffer_size, 
                                              alpha=self.alpha, 
                                              beta=self.beta, 
                                              epsilon=self.td_error_epsilon,
                                              device=device)

        # Initialize step counter
        self.steps_done = 0
        self.episodes_done = 0
    
    def select_action(self, state, training=True):
        """Select an action given the current state"

        Args:
            state: Current state
            training: Whether the agent is training or evaluating

        Returns:
            Selected action
        """
        if training:
            self.steps_done += 1
            
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state = state.unsqueeze(0).to(self.device) if state.dim() == 3 else state.to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the n-step buffer and replay buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """

        # Add transition to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if done:
            # Add n-step transitions to replay buffer
            while len(self.n_step_buffer) > 0:
                self._process_n_step_transition()
            return
        
        # if n-step buffer is not full, wait until it is
        if len(self.n_step_buffer) < self.n_steps:
            return
        
        # Process the oldest n-step transition
        self._process_n_step_transition()
    
    def _process_n_step_transition(self):
        """
        Process the oldest n-step transition in the buffer
        """
        if len(self.n_step_buffer) == 0:
            return
        
        # Get the oldest transition
        state, action, reward, next_state, done = self.n_step_buffer[0]

        # Calculate n-step return
        n_step_reward = reward

        # Look ahead through future transitions
        for i in range(1, len(self.n_step_buffer)):
            _, _, r, _, d = self.n_step_buffer[i]

            # Add discounted future reward
            n_step_reward += (self.gamma ** i) * r

            # If any future state is terminal, use that as the end point
            if d:
                done = True
                next_state = None
                break
        
        # If we had enough steps and didn't end, use the last state
        if not done and len(self.n_step_buffer) >= self.n_steps:
            next_state = self.n_step_buffer[-1][3]  # [3] is the next_state
        
        # Add transition to replay buffer
        self.memory.push(state, action, n_step_reward, next_state, done)

        # Remove the oldest transition
        self.n_step_buffer.popleft()


    
    def train(self, experience=None):
        """Train the agent on a batch of experiences

        
        Args:
            experience: Batch of experiences
            
        Returns:
            Dictionary of training metrics
        """
        # Skip training if no experience is provided
        if len(self.memory) < self.batch_size:
            return { "loss": None }
        
        # Sample batch of experiences
        if experience is None:
            batch = self.memory.sample(self.batch_size)
            if batch is None:
                return { "loss": None }
            state, action, reward, non_final_next_states, non_final_mask, done, weights, indices = batch
        else:
            state, action, reward, non_final_next_states, non_final_mask, done, weights, indices = experience
        
        # Calculate Q values
        state_action_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        
        # Compute target values
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            # Get actions from policy network
            next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            
            # Evaluate actions using target network
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze(1)

        # Compute expected Q values
        # For n-step learning, we apply the discount factor to the power of n
        expected_state_action_values = reward + ((self.gamma ** self.n_steps) * next_state_values * ~done)

        # Compute loss (Huber loss)
        element_wise_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction="none")

        # Apply importance sampling weights (weight the updates by the experience)
        loss = (element_wise_loss * weights).mean()

        # Update priorities
        td_errors = state_action_values - expected_state_action_values

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip Gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        self.policy_net.reset_noise()

        # Update priorities
        if indices is not None:
            self.memory.update_priorities(indices, td_errors)

        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Anneal beta
        if hasattr(self.memory, 'beta'):
            progress = min(1.0, self.steps_done / self.beta_annealing)
            self.memory.beta = self.beta + progress * (1.0 - self.beta)

        with torch.no_grad():
            # TD error metrics
            max_td_error = td_errors.abs().max().item()
            mean_td_error = td_errors.abs().mean().item()
            
            # PER weight variation
            weight_variance = weights.var().item() if weights is not None else 0

        # Get noisy network metrics
        noise_metrics = self._compute_noise_metrics() if self.steps_done % 1000 == 0 else {
            "weight_sigma": None, "bias_sigma": None, "noise_magnitude": None
        }
        return {
            "loss": loss.item(),
            "mean_q": state_action_values.mean().item(),  
            "td_error": td_errors.abs().mean().item(),
            "max_td_error": max_td_error,
            "mean_td_error": mean_td_error,
            "weight_variance": weight_variance,
            "memory_size": len(self.memory),
            "beta": getattr(self.memory, 'beta', None),
            "weight_sigma": noise_metrics["weight_sigma"],
            "bias_sigma": noise_metrics["bias_sigma"],
            "noise_magnitude": noise_metrics["noise_magnitude"],
            "network_distance": self._compute_network_distance() if self.steps_done % 1000 == 0 else None
        }
    def _compute_network_distance(self):
        """Compute L1 distance between policy and target networks"""
        with torch.no_grad():
            total_diff = 0.0
            param_count = 0
            for param_policy, param_target in zip(
                    self.policy_net.parameters(), self.target_net.parameters()):
                total_diff += torch.sum(torch.abs(param_policy - param_target)).item()
                param_count += param_policy.numel()
                
            return total_diff / param_count if param_count > 0 else 0.0
    def _compute_noise_metrics(self):
        """Compute metrics related to the noisy layers"""
        with torch.no_grad():
            weight_sigmas = []
            bias_sigmas = []
            noise_magnitudes = []
            
            # Collect metrics from all noisy layers
            for module in self.policy_net.modules():
                if isinstance(module, NoisyLinear):
                    # Average sigma values
                    weight_sigmas.append(module.weight_sigma.abs().mean().item())
                    bias_sigmas.append(module.bias_sigma.abs().mean().item())
                    
                    # Noise magnitude (average absolute noise)
                    if self.training:
                        weight_noise = (module.weight_sigma * module.weight_epsilon).abs().mean().item()
                        bias_noise = (module.bias_sigma * module.bias_epsilon).abs().mean().item()
                        noise_magnitudes.append((weight_noise + bias_noise) / 2)
            
            # Return averages across all layers
            return {
                "weight_sigma": np.mean(weight_sigmas) if weight_sigmas else 0.0,
                "bias_sigma": np.mean(bias_sigmas) if bias_sigmas else 0.0,
                "noise_magnitude": np.mean(noise_magnitudes) if noise_magnitudes else 0.0
            }
    
    def save(self, path):
        """Save agent state to the specified path"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
        }, path)
        
    def load(self, path):
        """Load agent state from the specified path"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No file found at {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episodes_done = checkpoint.get('episodes_done', 0)
