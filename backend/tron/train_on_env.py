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


env = TronBaseEnvTwoPlayer()
# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

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

def create_agent(state_dim, n_actions):
    current_net = Tron_DQN(state_dim, n_actions).to(device)
    target_net = Tron_DQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(current_net.state_dict())
    
    return {
        "currentNET": current_net,
        "targetNET": target_net,
        "optimizer": optim.Adam(current_net.parameters(), lr=LR),
        "memory": deque(maxlen=10000)
    }

# Create agents
agent1 = create_agent(state_dim, n_actions)
agent2 = create_agent(state_dim, n_actions)
AGENTS = [agent1, agent2]

def select_action(agent, state):
    if random.random() < agent["epsilon"]:
        return random.randint(0, 2)  # 0, 1, or 2
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = agent["currentNET"](state_tensor)
        return q_values.argmax().item()
    
def store_transition(agent, state, action, reward, next_state, done):
    agent["memory"].append((state, action, reward, next_state, done))

def sample_batch(agent):
    batch = random.sample(agent["memory"], BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    return (
        torch.FloatTensor(states).to(device),
        torch.LongTensor(actions).to(device),
        torch.FloatTensor(rewards).to(device),
        torch.FloatTensor(next_states).to(device),
        torch.FloatTensor(dones).to(device)
    )

# ---- TRAINING ----- #
num_episodes = 1000
epsilon_start = 1.0
epsilon_decay = 0.995 # Increase for less reward in later episodes
epsilon_end = 0.01

for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device)
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
    done = False

    while not done:
        # --- Selecting Action --- #
        actionAgent1 = select_action(state[0], epsilon, agent1)
        actionAgent2 = select_action(state[1], epsilon, agent2)

        # --- Take the Action --- #
        next_state, rewards, done, _, _ = env.step([actionAgent1, actionAgent2])

        next_state = torch.tensor 

