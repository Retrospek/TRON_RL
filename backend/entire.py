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
            "agent1": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=float),
            "agent2": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=float),
            "agent1_direction": spaces.Discrete(4),
            "agent2_direction": spaces.Discrete(4),
            "board": spaces.Box(low=0, high=2, shape=(self.size, self.size), dtype=int)
        })

        self.action_space = spaces.Discrete(4)
        self.reset()

    def reset(self):
        self.agent1_pos = np.array([self.size // 4, self.size // 2], dtype=float)
        self.agent2_pos = np.array([3 * self.size // 4, self.size // 2], dtype=float)

        self.agent1_direction = 1  # Initially moving right
        self.agent2_direction = 3  # Initially moving left

        self.board = np.zeros((self.size, self.size), dtype=int)
        # Only mark initial positions if they align with grid cells
        self.board[int(self.agent1_pos[1]), int(self.agent1_pos[0])] = 1
        self.board[int(self.agent2_pos[1]), int(self.agent2_pos[0])] = 2

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
        self.steps_taken += 1

        # Move agents and get validity of moves
        agent1_move_valid = self._move_agent(1)
        agent2_move_valid = self._move_agent(2)

        # Check if either agent made an invalid move or if the game is over
        self.done = not (agent1_move_valid and agent2_move_valid)

        # Rewards and Penalties
        rewards = [0, 0]
        if self.done:
            if not agent1_move_valid and not agent2_move_valid:
                rewards = [-1, -1]  # Both lose
            elif not agent1_move_valid:
                rewards = [-1, 1]   # Agent 1 loses, Agent 2 wins
            elif not agent2_move_valid:
                rewards = [1, -1]   # Agent 1 wins, Agent 2 loses
        else:
            rewards = [0.1, 0.1]  # Small positive reward for surviving

        return {
            "agent1": self.agent1_pos,
            "agent2": self.agent2_pos,
            "agent1_direction": self.agent1_direction,
            "agent2_direction": self.agent2_direction,
            "board": self.board
        }, rewards, [self.done, self.done], [False, False], {"steps_taken": self.steps_taken}

    def _move_agent(self, agent_id):
        """Move the agent in continuous space based on its direction."""
        # Get current position and direction
        pos = self.agent1_pos if agent_id == 1 else self.agent2_pos
        direction = self.agent1_direction if agent_id == 1 else self.agent2_direction
        trail_value = 1 if agent_id == 1 else 2

        # Calculate movement vector based on direction
        move_vector = {
            0: np.array([0, -self.move_speed]),  # Up
            1: np.array([self.move_speed, 0]),   # Right
            2: np.array([0, self.move_speed]),   # Down
            3: np.array([-self.move_speed, 0])   # Left
        }[direction]

        # Calculate next position
        next_pos = pos + move_vector

        # Get current and next grid positions
        current_grid_pos = np.floor(pos).astype(int)
        next_grid_pos = np.floor(next_pos).astype(int)

        # Check bounds
        if (next_grid_pos[0] < 0 or next_grid_pos[0] >= self.size or
            next_grid_pos[1] < 0 or next_grid_pos[1] >= self.size):
            return False

        # Check if we're entering a new grid cell
        if not np.array_equal(current_grid_pos, next_grid_pos):
            # Check if new cell is occupied
            if self.board[next_grid_pos[1], next_grid_pos[0]] != 0:
                return False
            # Mark the new cell
            self.board[next_grid_pos[1], next_grid_pos[0]] = trail_value

        # Update position
        if agent_id == 1:
            self.agent1_pos = next_pos
        else:
            self.agent2_pos = next_pos

        return True

    def render(self, mode='human'):
        """Render the current state of the environment using pygame."""
        if mode == 'human':
            if not hasattr(self, "screen"):
                pygame.init()
                self.screen = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("Tron Lightcycle Game")

            self.screen.fill((0, 0, 0))
            grid_size = self.window_size // self.size

            # Draw the trails
            for y in range(self.size):
                for x in range(self.size):
                    cell_value = self.board[y, x]
                    if cell_value == 1:
                        pygame.draw.rect(self.screen, (0, 255, 255), 
                                      (x * grid_size, y * grid_size, grid_size, grid_size))
                    elif cell_value == 2:
                        pygame.draw.rect(self.screen, (255, 0, 255), 
                                      (x * grid_size, y * grid_size, grid_size, grid_size))
            
            # Draw the agents
            pygame.draw.circle(self.screen, (0, 255, 255),
                            (int(self.agent1_pos[0] * grid_size + grid_size // 2),
                             int(self.agent1_pos[1] * grid_size + grid_size // 2)),
                            grid_size // 2)
            pygame.draw.circle(self.screen, (255, 0, 255),
                            (int(self.agent2_pos[0] * grid_size + grid_size // 2),
                             int(self.agent2_pos[1] * grid_size + grid_size // 2)),
                            grid_size // 2)

            pygame.display.flip()