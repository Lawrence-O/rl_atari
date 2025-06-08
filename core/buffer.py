import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    """Simple replay buffer for storing and sampling experiences"""
    
    def __init__(self, capacity, device="cpu"):
        """Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
            device: Device to store tensors on (only used during sampling)
        """
        self.buffer = deque(maxlen=capacity)
        self.device = device
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to the buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Ensure state is on CPU
        if isinstance(state, torch.Tensor) and state.device.type != 'cpu':
            state = state.detach().cpu()
            
        # Ensure next_state is on CPU (if not None)
        if next_state is not None and isinstance(next_state, torch.Tensor) and next_state.device.type != 'cpu':
            next_state = next_state.detach().cpu()
        
        # Convert scalars for memory efficiency
        if isinstance(action, torch.Tensor):
            action = action.item()
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        if isinstance(done, torch.Tensor):
            done = done.item()
            
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Sample a batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of batched experiences (state, action, reward, next_state, done)
            Only at this point are tensors moved to the specified device
        """
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # Convert to torch tensors and move to device only during sampling
        state = torch.cat([s.unsqueeze(0) if len(s.shape) == 3 else s for s in state]).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        
        # Handle different types of next_state (could be None if episode ended)
        non_final_mask = torch.tensor([s is not None for s in next_state], 
                                      dtype=torch.bool).to(self.device)
        
        # Only create tensor for non-final states
        if any(non_final_mask):
            non_final_next_states = torch.cat([
                s.unsqueeze(0) if len(s.shape) == 3 else s 
                for s in next_state if s is not None
            ]).to(self.device)
        else:
            # Create empty tensor if all states are final
            non_final_next_states = torch.zeros((0,) + state.shape[1:], device=self.device)
        
        done = torch.tensor(done, dtype=torch.bool).to(self.device)
        
        return state, action, reward, non_final_next_states, non_final_mask, done
    
    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)

class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with prioritized experience replay"""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=1e-6, device="cpu"):
        """Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Weight to assign to priority sampling
            beta: Weight to assign to importance sampling
            device: Device to store tensors on (only used during sampling)
        """
        super(PrioritizedReplayBuffer, self).__init__(capacity, device)
        self.alpha = alpha
        self.beta = beta
        self.td_error_epsilon = float(epsilon)
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to the buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        super().push(state, action, reward, next_state, done)
        self.priorities[len(self.buffer) - 1] = self.max_priority
        
    def sample(self, batch_size):
        """Sample a batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of batched experiences (state, action, reward, next_state, done, weights, indices)
            Only at this point are tensors moved to the specified device
        """
        if len(self.buffer) < batch_size:
            return None
        
        # Compute priorities and weights
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        
        # Sample experiences
        indices = torch.multinomial(probs, batch_size, replacement=True)

        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** -self.beta
        weights /= weights.max()

        # Get the batch
        batch = [self.buffer[i] for i in indices]
        state, action, reward, next_state, done = zip(*batch)
        
        # Convert to torch tensors and move to device only during sampling
        state = torch.cat([s.unsqueeze(0) if len(s.shape) == 3 else s for s in state]).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.bool).to(self.device)
        
        # Handle different types of next_state (could be None if episode ended)
        non_final_mask = torch.tensor([s is not None for s in next_state], 
                                      dtype=torch.bool).to(self.device)
        
        # Only create tensor for non-final states
        if any(non_final_mask):
            non_final_next_states = torch.cat([
                s.unsqueeze(0) if len(s.shape) == 3 else s 
                for s in next_state if s is not None
            ]).to(self.device)
        else:
            # Create empty tensor if all states are final
            non_final_next_states = torch.zeros((0,) + state.shape[1:], device=self.device)
        
        return state, action, reward, non_final_next_states, non_final_mask, done, weights, indices
    def update_priorities(self, indices, td_errors):
        """Update priorities for sampled experiences"

        Args:
            indices: Indices of sampled experiences
            td_errors: TD errors for each experience
        """
        # Make sure indices and priorities are on the same device
        device = self.priorities.device
        
        # Calculate absolute TD errors and keep on the same device
        abs_td_errors = torch.abs(td_errors).detach()
        new_priorities = (abs_td_errors + self.td_error_epsilon)
        
        # Make sure indices is on the same device as priorities
        if indices.device != device:
            indices = indices.to(device)
        
        # Make sure new_priorities is on the same device as priorities
        if new_priorities.device != device:
            new_priorities = new_priorities.to(device)
            
        # Now both indices and new_priorities are on the same device as self.priorities
        self.priorities[indices] = new_priorities
        self.max_priority = max(self.max_priority, new_priorities.max().item())