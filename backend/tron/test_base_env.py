import gym
from base import TronBaseEnvTwoPlayer  # Assuming the environment is saved in TronBaseEnvTwoPlayer.py

# Create the environment
env = TronBaseEnvTwoPlayer()

# Reset the environment to start a new episode
state, _ = env.reset()

# Print initial states of both agents
# print("Initial state:")
# print(f"Agent 1 State: {state[0]}")
# print(f"Agent 2 State: {state[1]}")

# Run a loop for a few steps (let's say 10 steps in this case)
for step in range(10):
    # Randomly choose actions for both agents (0 = Left, 1 = Right, 2 = Forward)
    #print(f"Step Number: {step}")
    actions = [env.action_space[0].sample(), env.action_space[1].sample()]
    
    # Step the environment forward with these actions
    next_state, rewards, done, truncated, info = env.step(actions)
    
    # Print the state and rewards after each step
    print(f"Step {step + 1}:")
    print(f"Agent 1 State: {next_state[0]}, Reward: {rewards[0]}")
    print(f"Agent 2 State: {next_state[1]}, Reward: {rewards[1]}")
    print(f"Game Info: {info}")
    
    # Render the environment (optional)
    env.render()

    # Check if the game is over
    if any(done):  # Game ends if any agent is done
        print("Game Over!")
        break

# Close the environment
#env.close()
