from collections import deque
import torch
import torch.nn as nn
import random
import gymnasium as gym
from PIL import Image
import numpy as np
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
from ale_py import ALEInterface


class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return torch.cat(state), action, reward, torch.cat(next_state), done

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(state_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        nn.init.orthogonal_(self.conv_layers[0].weight, np.sqrt(2))
        nn.init.orthogonal_(self.conv_layers[2].weight, np.sqrt(2))
        nn.init.orthogonal_(self.conv_layers[4].weight, np.sqrt(2))
        nn.init.orthogonal_(self.fc_layers[0].weight, np.sqrt(2))
        nn.init.orthogonal_(self.fc_layers[2].weight, np.sqrt(2))
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class DQNAgent:
    def __init__(self, env_name, gamma=0.99, epsilon=0.001,epsilon_decay=0.995, lr=1e-4):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.stack_frames_dim = 4
        self.frame_stack = None
        #self.state_dim = self.env.observation_space.shape[0]
        self.state_dim = self.stack_frames_dim
        self.action_dim = self.env.action_space.n
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_max = epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer()
        
    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return torch.argmax(self.model(state)).item()
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def preprocess(self, frame):
        frame = Image.fromarray(frame)
        frame = frame.convert('L')
        frame = frame.resize((84, 84))
        frame = np.array(frame) / 255.0
        frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        return frame
    
    def stack_frames(self, frame, new_episode=False):
        frame = self.preprocess(frame)
        if new_episode:
            self.frame_stack = torch.cat([frame]*self.stack_frames_dim, dim=1)
        else:
            self.frame_stack = torch.cat([self.frame_stack[:, 1:], frame], dim=1)
        return self.frame_stack
        
        
    def train(self, batch_size=32):
        if len(self.memory.buffer) < batch_size:
            return
        # Sample a batch of experiences
        state, action, reward, next_state, done = self.memory.sample(batch_size)

        # Convert to tensors
        state = state.to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = next_state.to(self.device)
        done = torch.tensor(done, dtype=torch.bool).to(self.device)

        # Compute loss
        q_values = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # next_q_values = self.target_model(next_state).max(1)[0] Normal DQN
            # Double DQN
            next_actions = self.model(next_state).max(1)[1].unsqueeze(1)
            next_q_values = self.target_model(next_state).gather(1, next_actions).squeeze(1)
            target = reward + self.gamma * next_q_values * ~done
        loss = nn.MSELoss()(q_values, target) 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def run(self, max_episodes=1000):
        episode_rewards = []
        for episode in range(max_episodes):
            state, _ = self.env.reset()
            state = self.stack_frames(state, new_episode=True)
            done = False
            total_reward = 0
            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.stack_frames(next_state)
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                loss = self.train()
            episode_rewards.append(total_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_max)
            self.update_target()
            print(f'Episode {episode+1}/{max_episodes}, Reward: {total_reward}, Loss: {loss}, Epsilon: {self.epsilon}')
        return episode_rewards
    
if __name__ == '__main__':
    env_name = 'ALE/Pong-v5'
    
    ale = ALEInterface()
    gym.register_envs(ale)

    # Create the agent
    agent = DQNAgent(env_name)
    # Run the agent
    rewards = agent.run(max_episodes=1500)  # Collect rewards
    
    # Plot the rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Learning Curve on Pong')
    plt.grid(True)
    
    # Plot the moving average
    window_size = 10
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r--', 
             label=f'{window_size}-episode Moving Average')
    plt.legend()
    plt.savefig('pong_dqn_performance.png')
    plt.show()
    
    # Run multiple evaluation episodes
    print("Evaluating trained agent across multiple episodes...")
    num_eval_episodes = 10
    eval_rewards = []
    best_reward = float('-inf')
    best_episode_states = []
    
    eval_env = gym.make(env_name)
    
    for i in range(num_eval_episodes):
        state, _ = eval_env.reset()
        state = agent.stack_frames(state, new_episode=True)
        done = False
        total_reward = 0
        episode_states = []
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            next_state = agent.stack_frames(next_state)
            
            # Store frame for possible replay
            if eval_env.render_mode == "rgb_array":
                episode_states.append(eval_env.render())
                
            state = next_state
            total_reward += reward
        
        eval_rewards.append(total_reward)
        print(f"Evaluation episode {i+1}/{num_eval_episodes}: Reward = {total_reward}")
        
        # Track best episode
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode_states = episode_states
    
    # Plot evaluation statistics
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, num_eval_episodes+1), eval_rewards)
    plt.axhline(y=np.mean(eval_rewards), color='r', linestyle='-', label=f'Mean: {np.mean(eval_rewards):.2f}')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Agent Evaluation on Pong')
    plt.legend()
    plt.savefig('pong_dqn_evaluation.png')
    
    # Record the best episode
    print(f"Recording the best episode (reward: {best_reward})...")
    record_env = gym.make(env_name, render_mode="rgb_array")
    record_env = RecordVideo(record_env, "videos", name_prefix="pong-dqn-best")
    
    state, _ = record_env.reset()
    state = agent.stack_frames(state, new_episode=True)
    done = False
    total_reward = 0
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = record_env.step(action)
        done = terminated or truncated
        next_state = agent.stack_frames(next_state)
        state = next_state
        total_reward += reward
    
    print(f"Best game recording saved with total reward: {total_reward}")
    record_env.close()
    print("Best game recording saved to the 'videos' directory")

