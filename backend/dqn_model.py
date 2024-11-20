import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNModel, self).__init__()
        
        # Define the neural network layers
        self.fc1 = nn.Linear(state_dim, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 128)        # Second hidden layer
        self.fc3 = nn.Linear(128, action_dim) # Output layer
        
    def forward(self, state):
        """Forward pass through the network."""
        x = F.relu(self.fc1(state))  # Apply ReLU activation on the first hidden layer
        x = F.relu(self.fc2(x))      # Apply ReLU activation on the second hidden layer
        return self.fc3(x)           # Return the Q-values for each action
