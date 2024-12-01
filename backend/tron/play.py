from train_on_env import Tron_DQN
import torch
import torch.nn as nn
import torch.optim as optim

n_actions = 3
input_size = 3

# Method 1: Loading just the model state
model1 = Tron_DQN(input_size, n_actions)
model1.load_state_dict(torch.load('tron_dqn_model.pth'))
model1.eval()

# Method 2: Loading model and optimizer state
checkpoint = torch.load('tron_dqn_checkpoint.pth')
model2 = Tron_DQN(input_size, n_actions)
model2.load_state_dict(checkpoint['model_state_dict'])

# Create an optimizer for the model before loading its state
optimizer = optim.Adam(model2.parameters(), lr=1e-4)  # Use the same learning rate as in training
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model2.eval()