import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import numpy as np

class CycleClashEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(CycleClashEnv, self).__init__()
        self.WIDTH = 1024
        self.HEIGHT = 720
        self.trail_thickness = 5
        self.speed = 5

        # 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
        self.action_space = spaces.MultiDiscrete([4, 4])  

        self.observation_space = spaces.Dict({
            "red_head": spaces.Box(low=0, high=max(self.WIDTH, self.HEIGHT), shape=(2,), dtype=np.float32),
            "blue_head": spaces.Box(low=0, high=max(self.WIDTH, self.HEIGHT), shape=(2,), dtype=np.float32),
            "red_trail": spaces.Box(low=0, high=max(self.WIDTH, self.HEIGHT), shape=(500, 2), dtype=np.float32),  
            "blue_trail": spaces.Box(low=0, high=max(self.WIDTH, self.HEIGHT), shape=(500, 2), dtype=np.float32),
        })

        self.red_head = [200.0, 400.0]
        self.blue_head = [800.0, 400.0]
        self.red_trail = [self.red_head[:]]
        self.blue_trail = [self.blue_head[:]]
        self.red_dir = [1, 0]
        self.blue_dir = [-1, 0]
        self.done = False

    def reset(self):
        self.red_head = [400.0, 400.0]
        self.blue_head = [600.0, 400.0]
        self.red_trail = [self.red_head[:]]
        self.blue_trail = [self.blue_head[:]]
        self.red_dir = [1, 0]
        self.blue_dir = [-1, 0]
        self.done = False
        return self._get_obs()

    def step(self, action):
        red_action, blue_action = action
        self._update_direction(red_action, blue_action)

        # Update positions
        self.red_head[0] += self.red_dir[0] * self.speed
        self.red_head[1] += self.red_dir[1] * self.speed
        self.red_trail.append(self.red_head[:])

        self.blue_head[0] += self.blue_dir[0] * self.speed
        self.blue_head[1] += self.blue_dir[1] * self.speed
        self.blue_trail.append(self.blue_head[:])

        red_collision = self._check_collision(self.red_head, self.red_trail[:-1]) or self._check_collision(self.red_head, self.blue_trail)
        blue_collision = self._check_collision(self.blue_head, self.blue_trail[:-1]) or self._check_collision(self.blue_head, self.red_trail)

        reward = 0
        if red_collision and blue_collision:
            self.done = True
            reward = -10  # Both lose
        elif red_collision:
            self.done = True
            reward = -10  # Red loses
        elif blue_collision:
            self.done = True
            reward = 10  # Blue loses
        else:
            reward = 1  # Survived this step

        return self._get_obs(), reward, self.done, {}

    def render(self, mode="human"):
        if mode == "human":
            pygame.init()
            screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            screen.fill((0, 0, 0))
            for i in range(1, len(self.red_trail)):
                pygame.draw.line(screen, (255, 0, 0), self.red_trail[i - 1], self.red_trail[i], self.trail_thickness)
            for i in range(1, len(self.blue_trail)):
                pygame.draw.line(screen, (0, 0, 255), self.blue_trail[i - 1], self.blue_trail[i], self.trail_thickness)
            pygame.draw.circle(screen, (255, 0, 0), (int(self.red_head[0]), int(self.red_head[1])), self.trail_thickness)
            pygame.draw.circle(screen, (0, 0, 255), (int(self.blue_head[0]), int(self.blue_head[1])), self.trail_thickness)
            pygame.display.flip()

    def _update_direction(self, red_action, blue_action):
        """Update player directions based on actions."""
        # Red player's direction
        if red_action == 0 and self.red_dir != [0, 1]:  # UP
            self.red_dir = [0, -1]
        elif red_action == 1 and self.red_dir != [0, -1]:  # DOWN
            self.red_dir = [0, 1]
        elif red_action == 2 and self.red_dir != [1, 0]:  # LEFT
            self.red_dir = [-1, 0]
        elif red_action == 3 and self.red_dir != [-1, 0]:  # RIGHT
            self.red_dir = [1, 0]

        # Blue player's direction
        if blue_action == 0 and self.blue_dir != [0, 1]:  # UP
            self.blue_dir = [0, -1]
        elif blue_action == 1 and self.blue_dir != [0, -1]:  # DOWN
            self.blue_dir = [0, 1]
        elif blue_action == 2 and self.blue_dir != [1, 0]:  # LEFT
            self.blue_dir = [-1, 0]
        elif blue_action == 3 and self.blue_dir != [-1, 0]:  # RIGHT
            self.blue_dir = [1, 0]

    def _check_collision(self, player_head, trail):
        if player_head[0] < 0 or player_head[0] >= self.WIDTH or player_head[1] < 0 or player_head[1] >= self.HEIGHT:
            return True
        for segment in trail:
            if abs(player_head[0] - segment[0]) < self.trail_thickness and abs(player_head[1] - segment[1]) < self.trail_thickness:
                return True
        return False

    def _get_obs(self):
        return {
            "red_head": np.array(self.red_head, dtype=np.float32),
            "blue_head": np.array(self.blue_head, dtype=np.float32),
            "red_trail": np.array(self.red_trail[-500:], dtype=np.float32),
            "blue_trail": np.array(self.blue_trail[-500:], dtype=np.float32),
        }
