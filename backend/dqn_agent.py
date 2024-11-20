import torch
import random
import numpy as np
from dqn_model import DQNModel  # Import the custom DQN model
from replay_buffer import ReplayBuffer  # Assume you have a ReplayBuffer class

class DQNAgent:
    def __init__(self, state_dim, action_dim, agent_id, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        self.epsilon = epsilon_start  # Exploration factor
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Initialize the Q-network and target network
        self.q_network = DQNModel(state_dim, action_dim)
        self.target_network = DQNModel(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Copy initial weights
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 64  # Size of batches used for training

    def select_action(self, state):
        """Select action based on epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(range(self.action_dim))  # Random action (exploration)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
            with torch.no_grad():
                q_values = self.q_network(state)  # Get Q-values from the model
            return torch.argmax(q_values).item()  # Choose action with the highest Q-value

    def train(self):
        """Train the DQN agent using experience replay."""
        if len(self.memory) < self.batch_size:
            return 0  # Not enough data to train
        
        # Sample a batch of experiences from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to torch tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute current Q-values for selected actions
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using the target network (next Q-values + rewards)
        next_q_values = self.target_network(next_states).max(1)[0]  # Max Q-value of next state
        target_q_values = rewards + (1 - dones) * 0.99 * next_q_values  # Bellman equation

        # Compute loss
        loss = F.mse_loss(q_values, target_q_values.detach())  # MSE loss

        # Perform gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, filepath):
        """Save the model parameters."""
        torch.save(self.q_network.state_dict(), filepath)

    def load(self, filepath):
        """Load the model parameters."""
        self.q_network.load_state_dict(torch.load(filepath))
