import os
import numpy as np
import torch
import h5py
import json
import time
from tqdm import tqdm
from core.environment import AtariWrapper, ClassicControlWrapper
from core.utils import set_seed

class DatasetCollector:
    """Collects transition data from environments for offline RL."""
    
    def __init__(self, env_name, agent=None, save_dir="datasets", device="cpu", 
                 frame_stack=4, epsilon=0.1, is_atari=True, seed=42):
        """Initialize the dataset collector.
        
        Args:
            env_name: Name of the environment
            agent: Optional agent for collecting data (if None, uses random policy)
            save_dir: Directory to save datasets
            device: Device to run the agent on
            frame_stack: Number of frames to stack (for Atari)
            epsilon: Exploration rate when using a trained agent (0 = greedy)
            is_atari: Whether the environment is an Atari game
            seed: Random seed for reproducibility
        """
        self.env_name = env_name
        self.agent = agent
        self.device = torch.device(device)
        self.epsilon = epsilon
        self.is_atari = is_atari
        
        # Create environment
        if is_atari:
            self.env = AtariWrapper(
                env_name=env_name,
                frame_skip=4,
                frame_stack=frame_stack,
                episode_life=True,
                clip_rewards=False  # Don't clip rewards for dataset collection
            )
        else:
            self.env = ClassicControlWrapper(env_name=env_name)
            
        # Create save directory
        game_name = env_name.split("/")[-1].split("-")[0]
        self.dataset_dir = os.path.join(save_dir, game_name)
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Set random seed
        set_seed(seed)
        
        # Dataset stats
        self.total_transitions = 0
        self.episodes = 0
        self.stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "total_transitions": 0,
            "environment": env_name,
            "agent_type": type(agent).__name__ if agent else "random",
            "epsilon": epsilon,
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def collect_random_data(self, num_transitions=1000000, max_episode_steps=108000):
        """Collect data using a random policy.
        
        Args:
            num_transitions: Number of transitions to collect
            max_episode_steps: Maximum steps per episode
            
        Returns:
            Path to the saved dataset
        """
        print(f"Collecting {num_transitions} transitions with random policy...")
        return self._collect_data(num_transitions, max_episode_steps, random_policy=True)
    
    def collect_agent_data(self, num_transitions=1000000, max_episode_steps=108000):
        """Collect data using the provided agent with epsilon-greedy policy.
        
        Args:
            num_transitions: Number of transitions to collect
            max_episode_steps: Maximum steps per episode
            
        Returns:
            Path to the saved dataset
        """
        if self.agent is None:
            raise ValueError("Agent must be provided to collect agent data")
            
        print(f"Collecting {num_transitions} transitions with agent (epsilon={self.epsilon})...")
        return self._collect_data(num_transitions, max_episode_steps, random_policy=False)
    
    def collect_mixed_data(self, num_transitions=1000000, max_episode_steps=108000, 
                          random_ratio=0.3):
        """Collect data using a mix of random policy and agent policy.
        
        Args:
            num_transitions: Number of transitions to collect
            max_episode_steps: Maximum steps per episode
            random_ratio: Ratio of episodes to collect with random policy
            
        Returns:
            Path to the saved dataset
        """
        if self.agent is None:
            raise ValueError("Agent must be provided to collect mixed data")
            
        print(f"Collecting {num_transitions} transitions with mixed policy (random ratio={random_ratio})...")
        
        # Number of transitions to collect with each policy
        random_transitions = int(num_transitions * random_ratio)
        agent_transitions = num_transitions - random_transitions
        
        # Collect random data
        if random_transitions > 0:
            self._collect_data(random_transitions, max_episode_steps, random_policy=True)
            
        # Collect agent data
        if agent_transitions > 0:
            self._collect_data(agent_transitions, max_episode_steps, random_policy=False)
            
        # Save dataset
        return self._save_dataset()
            
    def _collect_data(self, num_transitions, max_episode_steps, random_policy=False):
        """Internal method to collect data from the environment.
        
        Args:
            num_transitions: Number of transitions to collect
            max_episode_steps: Maximum steps per episode
            random_policy: Whether to use a random policy
            
        Returns:
            Path to the saved dataset
        """
        # Initialize buffers for batch storage
        states_buffer = []
        actions_buffer = []
        rewards_buffer = []
        next_states_buffer = []
        dones_buffer = []
        
        # Initialize counters
        transitions_collected = 0
        episode_reward = 0
        episode_steps = 0
        
        # Progress bar
        pbar = tqdm(total=num_transitions)
        
        # Start data collection
        state, _ = self.env.reset()
        
        # Preprocess observation if using an agent
        if not random_policy and self.agent is not None:
            state = self._preprocess_observation(state)
        
        # Main collection loop
        while transitions_collected < num_transitions:
            # Select action
            if random_policy:
                # Random action
                action = self.env.action_space.sample()
            else:
                # Agent action
                with torch.no_grad():
                    action = self.agent.select_action(state, training=False)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store pre-processed observation
            if not random_policy and self.agent is not None:
                states_buffer.append(state.cpu().numpy())
            else:
                states_buffer.append(state)
                
            actions_buffer.append(action)
            rewards_buffer.append(reward)
            
            # Store raw next_state (will be preprocessed before using)
            next_states_buffer.append(next_state)
            dones_buffer.append(done)
            
            # Update episode statistics
            episode_reward += reward
            episode_steps += 1
            transitions_collected += 1
            pbar.update(1)
            
            # Move to next state
            if not done:
                state = next_state
                if not random_policy and self.agent is not None:
                    state = self._preprocess_observation(state)
            else:
                # End of episode, reset
                self.episodes += 1
                self.stats["episode_rewards"].append(episode_reward)
                self.stats["episode_lengths"].append(episode_steps)
                
                # Print episode results
                pbar.set_description(f"Episode {self.episodes}: Reward={episode_reward:.2f}, Steps={episode_steps}")
                
                # Reset counters
                episode_reward = 0
                episode_steps = 0
                
                # Reset environment
                state, _ = self.env.reset()
                if not random_policy and self.agent is not None:
                    state = self._preprocess_observation(state)
                    
            # Save batch if buffer gets too large
            if len(states_buffer) >= 10000:
                self._save_batch(states_buffer, actions_buffer, rewards_buffer, 
                               next_states_buffer, dones_buffer)
                
                # Clear buffers
                states_buffer = []
                actions_buffer = []
                rewards_buffer = []
                next_states_buffer = []
                dones_buffer = []
        
        # Save final batch
        if states_buffer:
            self._save_batch(states_buffer, actions_buffer, rewards_buffer, 
                           next_states_buffer, dones_buffer)
        
        # Close progress bar
        pbar.close()
        
        # Update statistics
        self.stats["total_transitions"] = self.total_transitions
        
        # Save dataset
        return self._save_dataset()
    
    def _preprocess_observation(self, observation):
        """Preprocess observation for agent input."""
        if observation is None:
            return None
            
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        
        # Handle different observation shapes
        if len(observation.shape) == 1:  # Vector observation (e.g., CartPole)
            # Add batch dimension for vector observations
            observation = observation.reshape(1, -1)
        elif len(observation.shape) == 3:  # Image observation (e.g., Atari)
            # Add batch dimension for image observations [C, H, W]
            observation = observation.reshape(1, *observation.shape)
        
        # Convert to torch tensor and move to device
        tensor = torch.FloatTensor(observation).to(self.device)
        return tensor
    
    def _save_batch(self, states, actions, rewards, next_states, dones):
        """Save a batch of transitions to temporary storage."""
        # Convert to numpy arrays
        if isinstance(states[0], torch.Tensor):
            # Handle tensors
            states = np.concatenate([s.cpu().numpy() for s in states])
        else:
            # Handle numpy arrays
            states = np.array(states)
            
        actions = np.array(actions)
        rewards = np.array(rewards)
        
        if isinstance(next_states[0], torch.Tensor):
            next_states = np.concatenate([s.cpu().numpy() for s in next_states])
        else:
            next_states = np.array(next_states)
            
        dones = np.array(dones)
        
        # Create temporary file path
        batch_file = os.path.join(self.dataset_dir, f"batch_{self.total_transitions}.npz")
        
        # Save batch
        np.savez_compressed(
            batch_file,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones
        )
        
        # Update total transitions
        self.total_transitions += len(states)
    
    def _save_dataset(self):
        """Combine all batches into a single dataset file."""
        # Get all batch files
        batch_files = sorted([
            os.path.join(self.dataset_dir, f) for f in os.listdir(self.dataset_dir)
            if f.startswith("batch_") and f.endswith(".npz")
        ])
        
        if not batch_files:
            print("No data batches found!")
            return None
        
        # Create dataset file path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dataset_file = os.path.join(
            self.dataset_dir, 
            f"{self.env_name.split('/')[-1]}_{self.total_transitions}_{timestamp}.h5"
        )
        
        # Create H5 file
        with h5py.File(dataset_file, 'w') as f:
            # Create dataset groups
            for i, batch_file in enumerate(batch_files):
                # Load batch
                batch = np.load(batch_file)
                
                # Create group for batch
                if i == 0:
                    # Create datasets with total size
                    total_size = self.total_transitions
                    state_shape = batch['states'].shape[1:]
                    action_shape = batch['actions'].shape[1:] if len(batch['actions'].shape) > 1 else ()
                    
                    # Create datasets
                    states_dset = f.create_dataset(
                        'states', 
                        shape=(total_size, *state_shape),
                        dtype=np.float32,
                        chunks=True
                    )
                    actions_dset = f.create_dataset(
                        'actions', 
                        shape=(total_size, *action_shape),
                        dtype=np.int32 if not action_shape else np.float32,
                        chunks=True
                    )
                    rewards_dset = f.create_dataset(
                        'rewards', 
                        shape=(total_size,),
                        dtype=np.float32,
                        chunks=True
                    )
                    next_states_dset = f.create_dataset(
                        'next_states', 
                        shape=(total_size, *state_shape),
                        dtype=np.float32,
                        chunks=True
                    )
                    dones_dset = f.create_dataset(
                        'dones', 
                        shape=(total_size,),
                        dtype=np.bool_,
                        chunks=True
                    )
                    
                    # Add offset
                    offset = 0
                
                # Get batch size
                batch_size = batch['states'].shape[0]
                
                # Add data to datasets
                states_dset[offset:offset+batch_size] = batch['states']
                actions_dset[offset:offset+batch_size] = batch['actions']
                rewards_dset[offset:offset+batch_size] = batch['rewards']
                next_states_dset[offset:offset+batch_size] = batch['next_states']
                dones_dset[offset:offset+batch_size] = batch['dones']
                
                # Update offset
                offset += batch_size
                
                # Remove batch file
                os.remove(batch_file)
            
            # Add metadata
            f.attrs['total_transitions'] = self.total_transitions
            f.attrs['env_name'] = self.env_name
            f.attrs['agent_type'] = self.stats['agent_type']
            f.attrs['epsilon'] = self.epsilon
            f.attrs['episodes'] = self.episodes
            f.attrs['creation_date'] = self.stats['creation_date']
            
            # Add episode stats
            if self.stats['episode_rewards']:
                f.create_dataset('episode_rewards', data=np.array(self.stats['episode_rewards']))
                f.create_dataset('episode_lengths', data=np.array(self.stats['episode_lengths']))
        
        # Save stats as JSON
        stats_file = dataset_file.replace('.h5', '_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=4)
            
        print(f"Dataset saved to {dataset_file}")
        print(f"Statistics saved to {stats_file}")
        return dataset_file

class DatasetLoader:
    """Utility class for loading and processing offline RL datasets."""
    
    def __init__(self, dataset_path, device="cpu"):
        """Initialize the dataset loader.
        
        Args:
            dataset_path: Path to the dataset file
            device: Device to load the data on
        """
        self.dataset_path = dataset_path
        self.device = torch.device(device)
        self.metadata = {}
        
        # Load dataset metadata
        with h5py.File(dataset_path, 'r') as f:
            for key in f.attrs:
                self.metadata[key] = f.attrs[key]
            
            # Get dataset size
            self.size = f['states'].shape[0]
            
            # Get observation and action shapes
            self.observation_shape = f['states'].shape[1:]
            if len(f['actions'].shape) > 1:
                self.action_shape = f['actions'].shape[1:]
                self.discrete_actions = False
            else:
                self.action_shape = ()
                self.discrete_actions = True
                self.action_dim = f.attrs.get('action_dim', 
                                    int(np.max(f['actions'][:min(1000, self.size)])) + 1)
    
    def get_batch(self, batch_size=32, random=True):
        """Get a batch of transitions from the dataset.
        
        Args:
            batch_size: Size of the batch
            random: Whether to sample randomly or sequentially
            
        Returns:
            Dictionary of tensors containing the batch
        """
        with h5py.File(self.dataset_path, 'r') as f:
            if random:
                # Sample random indices
                indices = np.random.randint(0, self.size, size=batch_size)
            else:
                # Get next sequential batch
                start_idx = self.batch_count * batch_size % self.size
                indices = np.arange(start_idx, min(start_idx + batch_size, self.size))
                if len(indices) < batch_size:
                    # Wrap around
                    indices = np.concatenate([indices, np.arange(0, batch_size - len(indices))])
                self.batch_count += 1
            
            # Get batch
            states = torch.FloatTensor(f['states'][indices]).to(self.device)
            actions = torch.tensor(f['actions'][indices], 
                                  dtype=torch.long if self.discrete_actions else torch.float32).to(self.device)
            rewards = torch.FloatTensor(f['rewards'][indices]).to(self.device)
            next_states = torch.FloatTensor(f['next_states'][indices]).to(self.device)
            dones = torch.BoolTensor(f['dones'][indices]).to(self.device)
            
            return {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_states,
                'dones': dones
            }
    
    def get_all_data(self, max_size=None):
        """Get all data from the dataset.
        
        Args:
            max_size: Maximum number of transitions to load (None for all)
            
        Returns:
            Dictionary of tensors containing all data
        """
        with h5py.File(self.dataset_path, 'r') as f:
            size = min(self.size, max_size) if max_size is not None else self.size
            
            # Get all data
            states = torch.FloatTensor(f['states'][:size]).to(self.device)
            actions = torch.tensor(f['actions'][:size], 
                                dtype=torch.long if self.discrete_actions else torch.float32).to(self.device)
            rewards = torch.FloatTensor(f['rewards'][:size]).to(self.device)
            next_states = torch.FloatTensor(f['next_states'][:size]).to(self.device)
            dones = torch.BoolTensor(f['dones'][:size]).to(self.device)
            
            return {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_states,
                'dones': dones
            }
    
    def print_stats(self):
        """Print dataset statistics."""
        print(f"Dataset: {self.dataset_path}")
        print(f"Size: {self.size} transitions")
        print(f"Environment: {self.metadata.get('env_name', 'Unknown')}")
        print(f"Agent type: {self.metadata.get('agent_type', 'Unknown')}")
        print(f"Observation shape: {self.observation_shape}")
        print(f"Action {'dimension' if self.discrete_actions else 'shape'}: "
              f"{self.action_dim if self.discrete_actions else self.action_shape}")
        print(f"Created on: {self.metadata.get('creation_date', 'Unknown')}")
        
        # Load episode stats if available
        try:
            with h5py.File(self.dataset_path, 'r') as f:
                if 'episode_rewards' in f:
                    rewards = f['episode_rewards'][:]
                    lengths = f['episode_lengths'][:]
                    print(f"Episodes: {len(rewards)}")
                    print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
                    print(f"Average length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
        except:
            pass