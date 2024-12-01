import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from base import TronBaseEnvTwoPlayer

# ------- We're Creating a SELF_PLAY AGENT that tries to beat itself ------- #
env = TronBaseEnvTwoPlayer() # Really one agent just playing against itself
# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class Tron_DQN(nn.Module):
    def __init__(self, input_size, action_space):
        super(Tron_DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Hyperparameters
n_actions = 3
state_dim = 3
LR = 1e-4
BATCH_SIZE = 64
GAMMA = 0.99
TARGET_UPDATE = 10

# Create agents
agent = Tron_DQN(state_dim, n_actions).to(device)
target_agent = Tron_DQN(state_dim, n_actions).to(device)
target_agent.load_state_dict(agent.state_dict())
optimizer = optim.Adam(agent.parameters(), lr=LR)
memory = ReplayMemory(10000)

def select_action(state, epsilon, agent):
    # --- Logic --- #
    # Uses epsilon greedy strategy to both explore and exploit actions
    if random.random() > epsilon: # Exploit
        with torch.no_grad():
            return agent(state.unsqueeze(0)).max(1)[1].view(1, 1) # Takes index of highest probability action and reshapes into 1x1 tensor
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long) # Explore if random value less than or equal to epsilon

def optimize_model(memory, agent, target_agent, optimizer):
    # --- Logic --- #
    # Samples batch from memory, calculates Q values, and updates model
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = agent(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_agent(non_final_next_states).max(1)[0]
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(agent.parameters(), 1)
    optimizer.step()

# Training loop
num_episodes = 20 # Number of "games"
epsilon_start = 1.0
epsilon_decay = 0.995
epsilon_end = 0.01

for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device)
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
    done = False
    total_reward = 0

    while not done:
        actionAgent1 = select_action(state[0], epsilon, agent)
        actionAgent2 = select_action(state[1], epsilon, agent)

        next_state, rewards, done, _, _ = env.step([actionAgent1.item(), actionAgent2.item()])
        #print("# --- #")
        #print(f"Next State: {next_state}\n Rewards: {rewards}\n Done: {done}\n")
        reward = torch.tensor(rewards, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        
        memory.push(state[0], actionAgent1, next_state[0], reward[0].unsqueeze(0))
        memory.push(state[1], actionAgent2, next_state[1], reward[1].unsqueeze(0))

        state = next_state
        total_reward += sum(rewards)
        try:
            optimize_model(memory, agent, target_agent, optimizer)
        except:
            pass
    if episode % TARGET_UPDATE == 0:
        target_agent.load_state_dict(agent.state_dict())
    print(f"\n# ----- EPISODE {episode} ----- #")
    print(f"# Total Reward: {total_reward}")
    print(f"Epsilon: {epsilon:.2f}")
    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

print(" ")
print('Complete')

# Save the trained agent
torch.save(agent.state_dict(), 'tron_self_play_agent.pth')