import gym 
from gym import spaces
import pygame
import numpy as np

class Tron2Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.size = 50 # size of square grid for TRON
        self.window_size = 512

        self.observation_space = spaces.Dict({
            "agent1": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=int),  # Position (x, y) for Agent 1
            "agent2": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=int),  # Position (x, y) for Agent 2
            "agent1_direction": spaces.Discrete(4),  # Direction for Agent 1 (0 = up, 1 = right, 2 = down, 3 = left)
            "agent2_direction": spaces.Discrete(4),  # Direction for Agent 2
            "board": spaces.Box(low=0, high=2, shape=(self.size, self.size), dtype=int)  # Grid to represent trails
        })

        self.action_space = spaces.Discrete(4)
        
        
        self.reset
    
    def reset(self):

        self.agent1_pos = np.array([self.size // 4, self.size // 2])  # 1/4 to the right of the edge
        self.agent2_pos = np.array([3 * self.size // 4, self.size // 2]) # 3/4 to the right of the edge

        self.agent1_direction = 1  # Initially moving right (1: right)
        self.agent2_direction = 3

        self.board = np.zeros((self.size, self.size), dtype=int)
        self.board[tuple(self.agent1_pos)] = 1
        self.board[tuple(self.agent2_pos)] = 2

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

        self.agent1_direction = action[0]
        self.agent2_direction = action[1]

        self._move_agent(1) # Moving Agent 1
        self._move_agent(2) # Moving Agent 2

        agent1_move_valid = self._move_agent(1)
        agent2_move_valid = self._move_agent(2)


        if not agent1_move_valid or not agent2_move_valid or self.steps_taken >= self.max_steps:
            self.done = True


        rewards = [1, 1] if not self.done else [0, 0]

        return {
            "agent1": self.agent1_pos,
            "agent2": self.agent2_pos,
            "agent1_direction": self.agent1_direction,
            "agent2_direction": self.agent2_direction,
            "board": self.board
        }, rewards, [self.done, self.done], [False, False], {"steps_taken": self.steps_taken}
    
    def _move_agent(self, agent_id):
        agent_pos = self.agent1_pos if agent_id == 1 else self.agent2_pos
        direction = self.agent1_direction if agent_id == 1 else self.agent2_direction
        agent_trail_value = 1 if agent_id == 1 else 2  # Trail Value for marking squares type shit

        if direction == 0:  # Up
            next_pos = agent_pos - np.array([0, 1])
        elif direction == 1:  # Right
            next_pos = agent_pos + np.array([1, 0])
        elif direction == 2:  # Down
            next_pos = agent_pos + np.array([0, 1])
        elif direction == 3:  # Left
            next_pos = agent_pos - np.array([1, 0])

        if (next_pos[0] < 0 or next_pos[0] >= self.size or 
        next_pos[1] < 0 or next_pos[1] >= self.size or
        self.board[tuple(next_pos)] == agent_trail_value or
        self.board[tuple(next_pos)] == (3 - agent_trail_value)):  
            return False  # Invalid Move

        # Move is valid, so update the agent's position and the board
        agent_pos[:] = next_pos
        self.board[tuple(next_pos)] = agent_trail_value  # Mark the board with the agent's trail

        return True
    
    def check_game_over(self):

        # Check if Agent 1's head collides with an obstacle (boundary, opponent's trail, or its own trail)
        if (self.agent1_pos[0] < 0 or self.agent1_pos[0] >= self.size or
            self.agent1_pos[1] < 0 or self.agent1_pos[1] >= self.size or
            self.board[tuple(self.agent1_pos)] == 2 or  # Agent 1's head hits Agent 2's trail
            self.board[tuple(self.agent1_pos)] == 1):  # Agent 1's head hits its own trail
            return True  # Game over for Agent 1

        # Check if Agent 2's head collides with an obstacle (boundary, opponent's trail, or its own trail)
        if (self.agent2_pos[0] < 0 or self.agent2_pos[0] >= self.size or
            self.agent2_pos[1] < 0 or self.agent2_pos[1] >= self.size or
            self.board[tuple(self.agent2_pos)] == 1 or  # Agent 2's head hits Agent 1's trail
            self.board[tuple(self.agent2_pos)] == 2):  # Agent 2's head hits its own trail
            return True  # Game over for Agent 2

        return False  # Game not over yet



