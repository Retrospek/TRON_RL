import gym
from gym import spaces
import pygame
import numpy as np

class Tron2Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.size = 50  # Size of the grid
        self.window_size = 512
        self.move_speed = 0.5  # Speed of agent movement in continuous space

        # Observation space
        self.observation_space = spaces.Dict({
            "agent1": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=float),  # Position (x, y) for Agent 1 (continuous)
            "agent2": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=float),  # Position (x, y) for Agent 2 (continuous)
            "agent1_direction": spaces.Discrete(4),  # Direction for Agent 1 (0 = up, 1 = right, 2 = down, 3 = left)
            "agent2_direction": spaces.Discrete(4),  # Direction for Agent 2
            "board": spaces.Box(low=0, high=2, shape=(self.size, self.size), dtype=int)  # Grid to represent trails
        })

        # Action space (still discrete, but agents can move in any of 4 directions)
        self.action_space = spaces.Discrete(4)
        self.reset()

    def reset(self):
        self.agent1_pos = np.array([self.size // 4, self.size // 2], dtype=float)  # Continuous position
        self.agent2_pos = np.array([3 * self.size // 4, self.size // 2], dtype=float)  # Continuous position

        self.agent1_direction = 1  # Initially moving right (1: right)
        self.agent2_direction = 3  # Initially moving left (3: left)

        self.board = np.zeros((self.size, self.size), dtype=int)
        self.board[tuple(self.agent1_pos.astype(int))] = 1  # Mark agent's initial position
        self.board[tuple(self.agent2_pos.astype(int))] = 2

        self.done = False
        self.steps_taken = 0

        return {
            "agent1": self.agent1_pos,
            "agent2": self.agent2_pos,
            "agent1_direction": self.agent1_direction,
            "agent2_direction": self.agent2_direction,
            "board": self.board
        }

    def step(self, action):
        """Take a step in the environment."""
        self.agent1_direction = action[0]
        self.agent2_direction = action[1]

        agent1_move_valid = self._move_agent(1)  # Move Agent 1
        agent2_move_valid = self._move_agent(2)  # Move Agent 2
        
        print("Agent 1 position:", self.agent1_pos)
        print("Agent 2 position:", self.agent2_pos)
        print("Board state:\n", self.board)

        # Check if either agent made an invalid move or if the game is over
        self.done = self.check_game_over()  # Check if the game is over based on agent's positions

        print("Game Over status:", self.done)

        # Rewards and Penalties
        rewards = [0, 0]  # Default rewards (if game over, no reward)
        if self.done:
            # If the game is over, reward the winner and penalize the loser
            if self.board[tuple(self.agent1_pos.astype(int))] == 1:
                rewards[0] = -1  # Agent 1 loses
                rewards[1] = 1   # Agent 2 wins
            elif self.board[tuple(self.agent2_pos.astype(int))] == 2:
                rewards[0] = 1   # Agent 1 wins
                rewards[1] = -1  # Agent 2 loses
        else:
            # Reward for valid move (keeping the game going)
            rewards[0] = 3 if agent1_move_valid else -6  # Agent 1 gets positive for valid, negative for invalid
            rewards[1] = 3 if agent2_move_valid else -6  # Agent 2 gets positive for valid, negative for invalid

        return {
            "agent1": self.agent1_pos,
            "agent2": self.agent2_pos,
            "agent1_direction": self.agent1_direction,
            "agent2_direction": self.agent2_direction,
            "board": self.board
        }, rewards, [self.done, self.done], [False, False], {"steps_taken": self.steps_taken}

    def _move_agent(self, agent_id):
        """Move the agent in continuous space based on its direction."""
        agent_pos = self.agent1_pos if agent_id == 1 else self.agent2_pos
        direction = self.agent1_direction if agent_id == 1 else self.agent2_direction
        agent_trail_value = 1 if agent_id == 1 else 2  # Trail Value for marking squares

        if direction == 0:  # Up
            next_pos = agent_pos - np.array([0, self.move_speed])
        elif direction == 1:  # Right
            next_pos = agent_pos + np.array([self.move_speed, 0])
        elif direction == 2:  # Down
            next_pos = agent_pos + np.array([0, self.move_speed])
        elif direction == 3:  # Left
            next_pos = agent_pos - np.array([self.move_speed, 0])

        # Snap next_pos to nearest grid integer values for collision checking
        next_pos_int = np.round(next_pos).astype(int)

        print(f"Agent {agent_id} Next Position (rounded): {next_pos_int}")

        # Check if the move is valid (bounds checking and trail collision)
        if (next_pos_int[0] < 0 or next_pos_int[0] >= self.size or 
            next_pos_int[1] < 0 or next_pos_int[1] >= self.size or
            self.board[tuple(next_pos_int)] == agent_trail_value or
            self.board[tuple(next_pos_int)] == (3 - agent_trail_value)):  
            print(f"Agent {agent_id} hit an obstacle or went out of bounds.")
            return False  # Invalid Move

        # Move is valid, so update the agent's position and the board
        agent_pos[:] = next_pos
        self.board[tuple(next_pos_int)] = agent_trail_value  # Mark the board with the agent's trail
        return True

    def check_game_over(self):
        """Check if either agent has collided with an obstacle (boundary or trail)."""
        # Check if Agent 1 collides with an obstacle
        agent1_pos_int = np.round(self.agent1_pos).astype(int)
        if (agent1_pos_int[0] < 0 or agent1_pos_int[0] >= self.size or
            agent1_pos_int[1] < 0 or agent1_pos_int[1] >= self.size or
            self.board[tuple(agent1_pos_int)] == 2 or  # Agent 1 hits Agent 2's trail
            self.board[tuple(agent1_pos_int)] == 1):  # Agent 1 hits its own trail
            print("Agent 1 hit an obstacle!")
            return True  # Game over for Agent 1

        # Check if Agent 2 collides with an obstacle
        agent2_pos_int = np.round(self.agent2_pos).astype(int)
        if (agent2_pos_int[0] < 0 or agent2_pos_int[0] >= self.size or
            agent2_pos_int[1] < 0 or agent2_pos_int[1] >= self.size or
            self.board[tuple(agent2_pos_int)] == 1 or  # Agent 2 hits Agent 1's trail
            self.board[tuple(agent2_pos_int)] == 2):  # Agent 2 hits its own trail
            print("Agent 2 hit an obstacle!")
            return True  # Game over for Agent 2

        return False  # Game not over yet


    def render(self, mode='human'):
        """Render the current state of the environment using pygame."""
        if mode == 'human':
            # Initialize pygame window if not already initialized
            if not hasattr(self, "screen"):
                pygame.init()
                self.screen = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("Tron Lightcycle Game")

            # Clear screen
            self.screen.fill((0, 0, 0))

            # Scale grid size to fit the window size
            grid_size = self.window_size // self.size

            # Draw the grid
            for y in range(self.size):
                for x in range(self.size):
                    cell_value = self.board[y, x]
                    if cell_value == 1:
                        pygame.draw.rect(self.screen, (0, 255, 255), (x * grid_size, y * grid_size, grid_size, grid_size))
                    elif cell_value == 2:
                        pygame.draw.rect(self.screen, (255, 0, 255), (x * grid_size, y * grid_size, grid_size, grid_size))
            
            # Draw the agents (head positions)
            pygame.draw.circle(self.screen, (0, 255, 255), 
                               (int(self.agent1_pos[0] * grid_size + grid_size // 2), 
                                int(self.agent1_pos[1] * grid_size + grid_size // 2)), 
                               grid_size // 2)
            pygame.draw.circle(self.screen, (255, 0, 255), 
                               (int(self.agent2_pos[0] * grid_size + grid_size // 2), 
                                int(self.agent2_pos[1] * grid_size + grid_size // 2)), 
                               grid_size // 2)

            # Update the display
            pygame.display.flip()