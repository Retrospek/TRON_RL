import gym
from gym import spaces
import pygame
import numpy as np


class TronEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init(self, render_mode=None, size=50):
        
        

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size-1, shape=(2,), dtype=int),
                "enemy": spaces.Box(0, size-1, shape=(2,), dtype=int)
                
            }
        )

        
        