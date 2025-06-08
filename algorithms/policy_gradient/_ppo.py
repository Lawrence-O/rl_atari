import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.agent import Agent
from core.networks import PolicyNetwork
from torch.distributions.categorical import Categorical


class PPOAgent(Agent):
    """Proximal Policy Optimization agent for Atari environments"""

    def __init__(self, env, config, device="cpu"):
        """Initialize the PPO agent"""
        super(PPOAgent, self).__init__(env, device)
        
        # Extract configuration
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 3e-4)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_range = config.get("clip_range", 0.2)
        self.n_steps = config.get("n_steps", 2048)
        self.n_epochs = config.get("n_epochs", 10)
        self.batch_size = config.get("batch_size", 64)
        self.ent_coef = config.get("ent_coef", 0.01)
        self.vf_coef = config.get("vf_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.hidden_dim = config.get("hidden_dim", 512)
        
        # Get state and action dimensions
        state_shape = env.get_state_shape()
        num_actions = env.action_space.n
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_shape, num_actions, self.hidden_dim, device).to(device)
        self.value_net = PolicyNetwork(state_shape, 1, self.hidden_dim, device).to(device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=self.lr
        )
        
        # Initialize buffer - simplified for standard environments
        self.buffer = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }
        
        # For tracking progress
        self.step_ptr = 0
        self.steps_done = 0
        self.episodes_seen = 0
        
        # Variables to store last action's data
        self._last_log_prob = None
        self._last_value = None
    
    def select_action(self, state, training=True):
        """Select an action based on current state"""
        with torch.no_grad():
            # Get action logits 
            action_logits = self.policy_net(state)
            
            # Convert to distribution
            probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(probs=probs)
            
            if training:
                # Sample action from distribution
                action = dist.sample().item()
                
                # Store log prob and value for buffer
                self._last_log_prob = dist.log_prob(torch.tensor([action], device=self.device)).item()
                self._last_value = self.value_net(state).squeeze().item()
            else:
                # During evaluation, take the most likely action
                action = torch.argmax(probs, dim=-1).item()
                
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in buffer"""
        # Add to buffer
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['logprobs'].append(self._last_log_prob)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.buffer['values'].append(self._last_value)
        
        self.step_ptr += 1
        self.steps_done += 1
        
        # If buffer is full or episode ended, update policy
        if self.step_ptr >= self.n_steps or done:
            if done:
                self.episodes_seen += 1
            
            # Only update if we have enough steps
            if self.step_ptr > 0:
                training_metrics = self.train()
                return training_metrics
        
        return None
    
    def train(self, experience=None):
        """Train the agent using PPO algorithm"""
        # Skip if buffer is empty
        if self.step_ptr == 0:
            return None
            
        # Convert buffer to tensors
        states = torch.cat(self.buffer['states']).to(self.device)
        actions = torch.tensor(self.buffer['actions'], dtype=torch.long).to(self.device)
        old_logprobs = torch.tensor(self.buffer['logprobs'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(self.buffer['rewards'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.buffer['dones'], dtype=torch.bool).to(self.device)
        values = torch.tensor(self.buffer['values'], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Calculate value for final state (or 0 if done)
            if len(self.buffer['states']) > 0 and not self.buffer['dones'][-1]:
                next_state = self.buffer['states'][-1]
                next_value = self.value_net(next_state).squeeze().item()
            else:
                next_value = 0.0
            
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            
            for t in reversed(range(self.step_ptr)):
                if t == self.step_ptr - 1:
                    nextnonterminal = ~dones[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = ~dones[t]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        # PPO UPDATE LOOP - KEEPING EXACTLY AS ORIGINAL
        for _ in range(self.n_epochs):
            # Generate random indices
            indices = torch.randperm(self.step_ptr, device=self.device)
            
            # Process in minibatches
            for start in range(0, self.step_ptr, self.batch_size):
                end = start + self.batch_size
                if end > self.step_ptr:
                    end = self.step_ptr
                minibatch_indices = indices[start:end]
                
                # Get minibatch data
                mb_states = states[minibatch_indices]
                mb_actions = actions[minibatch_indices]
                mb_old_logprobs = old_logprobs[minibatch_indices]
                mb_returns = returns[minibatch_indices]
                mb_advantages = advantages[minibatch_indices]
                
                # Get current policy outputs
                mb_action_logits = self.policy_net(mb_states)
                mb_values = self.value_net(mb_states).squeeze()
                
                # Calculate new log probs and entropy
                probs = F.softmax(mb_action_logits, dim=-1)
                dist = Categorical(probs=probs)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # Calculate policy loss with clipping
                logratio = new_logprobs - mb_old_logprobs
                ratio = logratio.exp()
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = F.mse_loss(mb_values, mb_returns)
                
                # Total loss
                loss = pg_loss - self.ent_coef * entropy + self.vf_coef * v_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # Store metrics
                policy_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy.item())
        
        # Store advantage values for plotting
        advantage_values = advantages.detach().cpu().numpy().tolist()
        
        # Reset buffer
        for key in self.buffer:
            self.buffer[key] = []
        self.step_ptr = 0
        
        # Return metrics
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses),
            'advantage_values': advantage_values,
            'policy_grad_norm': self.max_grad_norm,
            'value_grad_norm': self.max_grad_norm,
            'steps_done': self.steps_done,
            'episodes_seen': self.episodes_seen
        }
    
    def save(self, path):
        """Save agent to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_seen': self.episodes_seen
        }, path)
    
    def load(self, path):
        """Load agent from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.episodes_seen = checkpoint.get('episodes_seen', 0)