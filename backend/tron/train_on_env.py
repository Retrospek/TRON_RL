import gymnasium as gym
import math
import random
import traceback
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque, namedtuple
from base import TronBaseEnvTwoPlayer

# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Deep Q-Network
class Tron_DQN(nn.Module):
    def __init__(self, input_size, action_space):
        super(Tron_DQN, self).__init__()
        
        # Dynamically determine input size
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Environment and device setup
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.998
TARGET_UPDATE = 50
MEMORY_SIZE = 10000
LR = 1e-4
NUM_EPISODES = 300

# Initialize environment
env = TronBaseEnvTwoPlayer()

# Determine input size
initial_state, _ = env.reset()
input_size = len(initial_state[0])
n_actions = 3  # Left, Right, Forward

# Initialize networks
policy_net = Tron_DQN(input_size, n_actions).to(device)
target_net = Tron_DQN(input_size, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Set to evaluation mode

# Optimizer and Memory
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_SIZE)

def select_action(state, eps_threshold):
    # Convert state to tensor if not already
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32, device=device)
    
    # Ensure state is 1D
    if state.dim() > 1:
        state = state.squeeze()
    
    # Epsilon-greedy action selection
    if random.random() > eps_threshold:
        with torch.no_grad():
            # Add batch dimension and get best action
            return policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 0
    
    # Sample batch
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    # Create tensors
    states = torch.tensor(batch.state, dtype=torch.float32, device=device)
    actions = torch.tensor(batch.action, dtype=torch.long, device=device)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    next_states = torch.tensor(batch.next_state, dtype=torch.float32, device=device)
    dones = torch.tensor(batch.done, dtype=torch.bool, device=device)
    
    # Current Q values
    current_q_values = policy_net(states).gather(1, actions)
    
    # Next Q values
    next_q_values = torch.zeros_like(rewards, device=device)
    with torch.no_grad():
        next_q_values[~dones] = target_net(next_states[~dones]).max(1)[0]
    
    # Compute expected Q values
    expected_q_values = rewards + (next_q_values * GAMMA)
    
    # Compute loss
    loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    optimizer.step()
    
    return loss.item()

# Training loop
print("Starting training...")
episode_rewards = []

for episode in range(NUM_EPISODES):
    # Decay epsilon
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY)
    
    # Reset environment
    state, _ = env.reset()
    state = [np.array(s) for s in state]
    
    total_reward = 0
    done = False
    
    while not done:
        # Select actions for both agents
        action1 = select_action(state[0], eps_threshold).item()
        action2 = select_action(state[1], eps_threshold).item()
        
        # Step environment
        next_state, rewards, done_list, _, _ = env.step([action1, action2])
        done = done_list[0]
        
        # Convert states to numpy for storage
        next_state = [np.array(s) for s in next_state]
        
        # Store transition in memory
        memory.push(
            state[0], [action1], 
            next_state[0], rewards[0], done
        )
        memory.push(
            state[1], [action2], 
            next_state[1], rewards[1], done
        )
        
        # Optimize model
        optimize_model()
        
        # Update state
        state = next_state
        total_reward += sum(rewards)
    
    # Update target network periodically
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    # Track rewards
    episode_rewards.append(total_reward)
    
    # Print progress
    if episode % 10 == 0:
        print(f"Episode {episode}: Total Reward = {total_reward}, Epsilon = {eps_threshold:.2f}")

print('Training Complete')

# Add this after training is complete
torch.save({
    'model_state_dict': policy_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'tron_dqn_checkpoint.pth')

# Optional: Plot learning curve
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(episode_rewards)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()