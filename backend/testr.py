import gym
import numpy as np
import random
from entire import Tron2Env

# Initialize the environment
env = Tron2Env()

# Reset the environment to initialize the agents and board
state = env.reset()

# Run a few steps in the environment
done = False
for _ in range(100):  # Limit to 100 steps to avoid infinite loop
    if done:
        break
    
    # Randomly select actions for Agent 1 and Agent 2 (just for testing)
    action1 = random.randint(0, 3)  # Random direction for Agent 1 (0=up, 1=right, 2=down, 3=left)
    action2 = random.randint(0, 3)  # Random direction for Agent 2
    
    # Step forward in the environment
    action = [action1, action2]
    state, rewards, done, _, _ = env.step(action)
    
    # Print the positions of the agents
    print(f"Agent 1 Position: {state['agent1']}, Agent 2 Position: {state['agent2']}")
    
    # Render the environment (this will pop up a window showing the game)
    env.render()

# Close the environment
env.close()