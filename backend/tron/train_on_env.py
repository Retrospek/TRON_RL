import gymnasium as gym
import math
import random
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from base import TronBaseEnvTwoPlayer

# Environment and device setup
env = TronBaseEnvTwoPlayer()
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# First, let's print out the state information to understand its structure
print("Environment Reset to understand state dimensions:")
initial_state, _ = env.reset()
print("Initial State Shape:", [s.shape for s in initial_state])
print("Initial State:", initial_state)

# Modify the DQN to dynamically adapt to input size
class Tron_DQN(nn.Module):
    def __init__(self, input_size, action_space):
        super(Tron_DQN, self).__init__()
        print(f"Initializing DQN with input size: {input_size}")
        
        # Dynamically determine input size
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_space)

    def forward(self, x):
        # Print input tensor details for debugging
        print(f"Forward pass input shape: {x.shape}")
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Determine actual input size dynamically
initial_state, _ = env.reset()
actual_state_dim = len(initial_state[0])  # Assuming first agent's state
print(f"Actual state dimension: {actual_state_dim}")

# Hyperparameters
n_actions = 3  # Assuming 3 possible actions
LR = 1e-4
BATCH_SIZE = 64
GAMMA = 0.99
TARGET_UPDATE = 10

# Create agents with dynamic input size
try:
    agent = Tron_DQN(actual_state_dim, n_actions).to(device)
    target_agent = Tron_DQN(actual_state_dim, n_actions).to(device)
    target_agent.load_state_dict(agent.state_dict())
    optimizer = optim.Adam(agent.parameters(), lr=LR)
except Exception as e:
    print("Error initializing agents:")
    traceback.print_exc()
    raise

def select_action(state, epsilon, agent):
    try:
        # Ensure state is a 1D tensor
        if len(state.shape) > 1:
            state = state.squeeze()
        
        # Ensure state is float tensor
        state = state.float()
        
        # Exploit or explore
        if random.random() > epsilon:
            with torch.no_grad():
                # Add batch dimension if missing
                if len(state.shape) == 1:
                    state = state.unsqueeze(0)
                return agent(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    except Exception as e:
        print("Error in select_action:")
        print(f"State shape: {state.shape}")
        traceback.print_exc()
        raise

# Rest of the training loop remains similar, with more robust state handling

# Modify the main training loop to handle state preprocessing
num_episodes = 10
epsilon_start = 1.0
epsilon_decay = 0.995
epsilon_end = 0.01

print("Starting training...")
for episode in range(num_episodes):
    try:
        # Reset environment and initial state
        state, _ = env.reset()
        
        # Convert to torch tensors explicitly
        state = [torch.tensor(s, dtype=torch.float32, device=device) for s in state]
        
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        done = False
        total_reward = 0

        while not done:
            # Select actions for both agents
            actionAgent1 = select_action(state[0], epsilon, agent)
            actionAgent2 = select_action(state[1], epsilon, agent)

            # Step environment
            next_state, rewards, done, _, _ = env.step([actionAgent1.item(), actionAgent2.item()])
            print(done)
            # Convert next state to torch tensors
            next_state = [torch.tensor(s, dtype=torch.float32, device=device) for s in next_state]
            rewards = torch.tensor(rewards, device=device)

            # Update state
            state = next_state
            total_reward += sum(rewards)

        print(f"\nEpisode {episode}: Total Reward = {total_reward}, Epsilon = {epsilon:.2f}")
    
    except Exception as e:
        print(f"Error in episode {episode}:")
        traceback.print_exc()

print('Training Complete')