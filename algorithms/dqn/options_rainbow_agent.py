from collections import deque
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from core.agent import Agent
from core.networks import *
from core.buffer import PrioritizedReplayBuffer


class OptionsRainbowAgent(Agent):
    """Rainbow agent extended with hierarchical options"""

    def __init__(self, env, config, device="cpu"):
        super(OptionsRainbowAgent, self).__init__(env, device)

        # Base Rainbow configuration
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 1e-4)
        self.batch_size = config.get("batch_size", 32)
        self.target_update_freq = config.get("target_update_freq", 10000)

        #Options configuration
        self.num_options = config.get("num_options", 4)
        self.option_epsilon = config.get("option_epsilon", 0.1)
        self.termination_regularization = config.get("termination_regularization", 0.01)
        self.meta_controller_lr = config.get("meta_controller_lr", 1e-3)
        self.option_lr = config.get("option_lr", 1e-3)

        # Prioritized replay buffer
        self.buffer_size = config.get("buffer_size", 100000)
        self.alpha = config.get("alpha", 0.6) # Priority exponent
        self.beta = config.get("beta", 0.4) # Importance sampling exponent
        self.td_error_epsilon = config.get("td_error_epsilon", 1e-6)
        self.beta_annealing = config.get("beta_annealing", 100000) # Steps to anneal beta to 1.0
        self.replay_start_size = config.get("replay_start_size", 80000) # Steps before starting to sample from replay buffer

        # Get state and action dimensions
        state_shape = env.get_state_shape()
        num_actions = env.action_space.n

        # N step learning
        self.n_steps = config.get("n_steps", 3)
        self.n_step_buffer = deque(maxlen=self.n_steps)

        # Intialize networks
        self.meta_controller = MetaController(state_shape, self.num_options, device=device).to(device)
        self.target_meta_controller = MetaController(state_shape, self.num_options, device=device).to(device)
        self.target_meta_controller.load_state_dict(self.meta_controller.state_dict())

        # Initialize option policies
        self.option_policies = nn.ModuleList([
            OptionPolicy(state_shape, num_actions, device=device).to(device) for _ in range(self.num_options)
        ])
        self.target_option_policies = nn.ModuleList([
            OptionPolicy(state_shape, num_actions, device=device).to(device) for _ in range(self.num_options)
        ])

        for i in range(self.num_options):
            self.target_option_policies[i].load_state_dict(self.option_policies[i].state_dict())
        
        # Initialize termination networks
        self.termination_network = TerminationNetwork(state_shape, self.num_options, device=device).to(device)

        self.meta_controller_optimizer = torch.optim.Adam(self.meta_controller.parameters(), lr=self.meta_controller_lr)
        self.option_optimizers = [
            torch.optim.Adam(policy.parameters(), lr=self.option_lr) for policy in self.option_policies
        ]
        self.termination_optimizer = torch.optim.Adam(self.termination_network.parameters(), lr=self.option_lr)
        # Initialize replay buffer
        self.meta_memory = PrioritizedReplayBuffer(
            capacity=self.buffer_size,
            alpha=self.alpha,
            beta=self.beta,
            epsilon=self.td_error_epsilon,
            device=device
        )

        self.options_memory = PrioritizedReplayBuffer(
            capacity=self.buffer_size,
            alpha=self.alpha,
            beta=self.beta,
            epsilon=self.td_error_epsilon,
            device=device
        )

        self.current_option = None
        self.option_start_state = None
        self.option_duration = 0
        self.option_return = 0.0
        self.steps_done = 0
        self.episodes_done = 0

        # Metrics
        self.option_frequencies = [0] * self.num_options
        self.option_durations_history = {i: [] for i in range(self.num_options)}
        self.option_returns_history = {i: [] for i in range(self.num_options)}
        self.termination_probs_history = {i: [] for i in range(self.num_options)}
        self.meta_q_values_history = []
        self.option_transition_matrix = np.zeros((self.num_options, self.num_options))
        self.previous_option = None

        random.seed(config.get("seed", 42))
        self.device = device
        
    def _track_option_selection(self, new_option):
        """Track option selection for analysis"""
        self.option_frequencies[new_option] += 1
        
        # Track transitions
        if self.previous_option is not None:
            self.option_transition_matrix[self.previous_option, new_option] += 1
        
        self.previous_option = new_option
    def _track_option_completion(self, option_id, duration, return_value):
        """Track completed option statistics"""
        self.option_durations_history[option_id].append(duration)
        self.option_returns_history[option_id].append(return_value)

    def _should_terminate_option(self, state):
        """Check if the current option should terminate"""
        if self.current_option is None:
            return True
        with torch.no_grad():
            termination_probs = self.termination_network(state)
            termination_prob = termination_probs[0, self.current_option].item()
        return random.random() < termination_prob
    
    def get_hierarchical_metrics(self):
        """Get current hierarchical learning metrics"""
        # Sample current termination probabilities
        if hasattr(self, 'termination_network'):
            with torch.no_grad():
                # Use a dummy state to get current termination probabilities
                if hasattr(self, 'option_start_state') and self.option_start_state is not None:
                    term_probs = self.termination_network(self.option_start_state)
                    for i in range(self.num_options):
                        prob = term_probs[0, i].item()
                        self.termination_probs_history[i].append(prob)
        
        return {
            "option_frequencies": self.option_frequencies.copy(),
            "option_durations": self.option_durations_history.copy(),
            "option_returns": self.option_returns_history.copy(),
            "termination_probabilities": self.termination_probs_history.copy(),
            "meta_q_values": self.meta_q_values_history.copy(),
            "option_transition_matrix": self.option_transition_matrix.copy()
        }


    def select_action(self, state, training=True):
        """Select action using current option or select new option"""
        if training:
            self.steps_done += 1
        
        if hasattr(self.meta_controller, 'reset_noise'):
            self.meta_controller.reset_noise()
        for policy in self.option_policies:
            if hasattr(policy, 'reset_noise'):
                policy.reset_noise()

        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state = state.unsqueeze(0).to(self.device) if state.dim() == 3 else state.to(self.device)
            
            if self.current_option is None or self._should_terminate_option(state):
                if self.current_option is not None:
                    self._store_option_transition(state)
                
                # Select new option using meta-controller
                self.current_option = self._select_option(state, training)
                self.option_start_state = state.clone()
                self.option_duration = 0
                self.option_return = 0.0
            
            # Execute current option policy
            option_policy = self.option_policies[self.current_option]
            action = option_policy(state).max(1)[1].item()

            return action
    
    def _select_option(self, state, training):
        """Select an option using the meta-controller"""
        if training and random.random() < self.option_epsilon:
            new_option = random.randint(0, self.num_options - 1)
        else:
            with torch.no_grad():
                option_values = self.meta_controller(state)
                self.meta_q_values_history.append(option_values.mean().item())
                new_option = option_values.max(1)[1].item()
            
        if training:
            self._track_option_selection(new_option)
        return new_option
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Update option statistics
        self.option_duration += 1
        self.option_return += reward

        # If the option has ended, process the n-step transitions
        if done:
            if self.current_option is not None:
                self._store_option_transition(next_state, done)
            
            while len(self.n_step_buffer) > 0:
                self._process_option_n_step_transition()
            self.current_option = None
            self.episodes_done += 1
        
        # If we have enough transitions in the n-step buffer, process them
        if len(self.n_step_buffer) >= self.n_steps:
            self._process_option_n_step_transition()
    def _process_option_n_step_transition(self):
        """Process n-step transitions and store in replay buffer"""
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
        self.options_memory.push(state, action, n_step_reward, next_state, done)

        # Remove the oldest transition
        self.n_step_buffer.popleft()
    
    def _store_option_transition(self, final_state, done=False):
        """Store the transition for the current option"""
        if self.option_start_state is not None and self.current_option is not None:
            self._track_option_completion(self.current_option, self.option_duration, self.option_return)
            self.meta_memory.push(
                state=self.option_start_state,
                action=self.current_option,
                reward=self.option_return,
                next_state=final_state if final_state is not None else None,
                done=done,
            )
    def train(self, experience=None):
        """Train meta-controller, option policies, and termination network"""
        if len(self.meta_memory) < self.replay_start_size:
            return {
                "buffer_size": len(self.meta_memory), 
                "buffer_target": self.replay_start_size,
                "loss": None,
                "hierarchical_metrics": self.get_hierarchical_metrics()
            }
        
        metrics = dict()

        # Train meta-controller
        if len(self.meta_memory) >= self.batch_size:
            meta_metrics = self._train_meta_controller()
            metrics.update(meta_metrics)
        
        # Train option policies
        option_losses = []
        for i in range(self.num_options):
            if len(self.options_memory) >= self.batch_size:
                option_loss = self._train_option_policy(i)
                option_losses.append(option_loss)
            else:
                option_losses.append(None)
        metrics["option_losses"] = option_losses
        
        # Train termination network
        if len(self.meta_memory) >= self.batch_size:
            termination_loss = self._train_termination_network()
            metrics["termination_loss"] = termination_loss
        
        # Update target networks
        if self.steps_done % self.target_update_freq == 0:
            self._update_target_networks()
        
        metrics["hierarchical_metrics"] = self.get_hierarchical_metrics()
        
        return metrics
    def _train_meta_controller(self):
        """Train the meta-controller using prioritized replay buffer"""
        batch = self.meta_memory.sample(self.batch_size)
        state, option, reward, non_final_next_states, non_final_mask, done, weights, indices = batch
        
        # Reset Noise
        if hasattr(self.meta_controller, 'reset_noise'):
            self.meta_controller.reset_noise()
            self.target_meta_controller.reset_noise()
        
        # Calculate Q values
        state_option_values = self.meta_controller(state).gather(1, option.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        next_option_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            # Get actions from policy network
            next_actions = self.meta_controller(non_final_next_states).max(1)[1].unsqueeze(1)
            
            # Evaluate actions using target network
            next_option_values[non_final_mask] = self.target_meta_controller(non_final_next_states).gather(1, next_actions).squeeze(1)
        
        # Compute Targets
        expected_option_values = reward + ((self.gamma ** self.option_duration) * next_option_values * ~done)

        # Compute loss (Huber loss)
        element_wise_loss = F.smooth_l1_loss(state_option_values, expected_option_values, reduction="none")

        # Apply importance sampling weights (weight the updates by the experience)
        loss = (element_wise_loss * weights).mean()

        # Optimize
        self.meta_controller_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_controller.parameters(), 10)
        self.meta_controller_optimizer.step()

        # Update priorities
        td_errors = state_option_values - expected_option_values
        if indices is not None:
            self.meta_memory.update_priorities(indices, td_errors)
        
        return {
            "meta_loss": loss.item(),
            "meta_q_values": state_option_values.mean().item(),
            "meta_td_error": td_errors.abs().mean().item()
        }
    
    def _train_option_policy(self, option_index):
        """Train a specific option policy using prioritized replay buffer"""
        batch = self.options_memory.sample(self.batch_size)
        state, action, reward, non_final_next_states, non_final_mask, done, weights, indices = batch
        
        # Reset Noise
        if hasattr(self.option_policies[option_index], 'reset_noise'):
            self.option_policies[option_index].reset_noise()
            self.target_option_policies[option_index].reset_noise()
        
        # Calculate Q values for the option policy
        state_action_values = self.option_policies[option_index](state).gather(1, action.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_actions = self.option_policies[option_index](non_final_next_states).max(1)[1].unsqueeze(1)
            next_state_values[non_final_mask] = self.target_option_policies[option_index](non_final_next_states).gather(1, next_actions).squeeze(1)
        
        # Compute Targets
        expected_state_action_values = reward + ((self.gamma ** self.n_steps) * next_state_values * ~done)

        # Compute loss (Huber loss)
        element_wise_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction="none")

        # Apply importance sampling weights (weight the updates by the experience)
        loss = (element_wise_loss * weights).mean()

        # Optimize
        self.option_optimizers[option_index].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.option_policies[option_index].parameters(), 10)
        self.option_optimizers[option_index].step()

        # Update priorities
        td_errors = state_action_values - expected_state_action_values
        if indices is not None:
            self.options_memory.update_priorities(indices, td_errors)
        
        return loss.item()
    
    def _train_termination_network(self):
        """Train the termination network"""
        batch = self.meta_memory.sample(self.batch_size)
        if batch is None:
            return None
        
        state, option, reward, non_final_next_states, non_final_mask, done, weights, indices = batch

        if hasattr(self.termination_network, 'reset_noise'):
            self.termination_network.reset_noise()
        
        # Get termination probabilities
        termination_probs = self.termination_network(state)
        
        # Target: terminate if next state is terminal or if option value is low
        next_option_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            current_option_values = self.meta_controller(state).gather(1, option.unsqueeze(1)).squeeze(1)
            if len(non_final_next_states) > 0:
                next_actions = self.meta_controller(non_final_next_states).max(1)[1].unsqueeze(1)
                next_option_values[non_final_mask] = self.target_meta_controller(non_final_next_states).gather(1, next_actions).squeeze(1)
            advantage = next_option_values - current_option_values

            termination_targets = torch.sigmoid(-advantage)
            
        
        # Extract relevant termination probabilities
        option_termination_probs = termination_probs.gather(1, option.unsqueeze(1)).squeeze(1)
        
        # Compute loss with regularization
        termination_loss = F.binary_cross_entropy(option_termination_probs, termination_targets, reduction='none')
        regularization = self.termination_regularization * termination_probs.mean(1)
        total_element_wise_loss = termination_loss + regularization
        loss = (total_element_wise_loss * weights).mean()
        
        # Optimize
        self.termination_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.termination_network.parameters(), 10)
        self.termination_optimizer.step()

        # if indices is not None:
        #     termination_td_error = torch.abs(option_termination_probs - termination_targets).detach()
        #     self.meta_memory.update_priorities(indices, termination_td_error)
        
        return loss.item()
    def _update_target_networks(self):
        """Update target networks"""
        self.target_meta_controller.load_state_dict(self.meta_controller.state_dict())
        for i in range(self.num_options):
            self.target_option_policies[i].load_state_dict(self.option_policies[i].state_dict())
    
    def save(self, path):
        """Save all networks"""
        torch.save({
            'meta_controller': self.meta_controller.state_dict(),
            'target_meta_controller': self.target_meta_controller.state_dict(),
            'option_policies': [policy.state_dict() for policy in self.option_policies],
            'target_option_policies': [policy.state_dict() for policy in self.target_option_policies],
            'termination_network': self.termination_network.state_dict(),
            'meta_optimizer': self.meta_controller_optimizer.state_dict(),
            'option_optimizers': [opt.state_dict() for opt in self.option_optimizers],
            'termination_optimizer': self.termination_optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'current_option': self.current_option
        }, path)
    
    def load(self, path):
        """Load all networks"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.meta_controller.load_state_dict(checkpoint['meta_controller'])
        self.target_meta_controller.load_state_dict(checkpoint['target_meta_controller'])
        
        for i, state_dict in enumerate(checkpoint['option_policies']):
            self.option_policies[i].load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint['target_option_policies']):
            self.target_option_policies[i].load_state_dict(state_dict)
            
        self.termination_network.load_state_dict(checkpoint['termination_network'])
        
        self.meta_controller_optimizer.load_state_dict(checkpoint['meta_optimizer'])
        for i, state_dict in enumerate(checkpoint['option_optimizers']):
            self.option_optimizers[i].load_state_dict(state_dict)
        self.termination_optimizer.load_state_dict(checkpoint['termination_optimizer'])
        
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        self.current_option = checkpoint.get('current_option', None)