import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np

class TronBaseEnvTwoPlayer(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(TronBaseEnvTwoPlayer, self).__init__()
        # ----------
        # Initializing the observation and action space
        # ----------
            # Going to define bounds, space, shape
                # Bound: The Board Size is limited
                # Space: Because we're playing the game tron we control a bike that can't move backward lmaoo obv
                # Shape: Shape of space and bound

        # ----- BOUNDS ----- # 

        self.board_width = 1024
        self.board_height = 768
        
        # ----- AGENTS ----- #

        self.speed = 15 # Increase for faster movement
        self.direction = [0, 180] # Angles where 0 is right and 180 is left

        self.agent_positions = [
            (int(self.board_width / 4), self.board_height / 2),
            (int(self.board_width - self.board_width / 4, self.board_height / 2))
        ]

        # Instantiating the starting positions of the trails and the sets that will hold each agents trails
        self.trails = [set(), set()]
        self.trails[0].add(self.agent_positions[0])
        self.trails[1].add(self.agent_positions[1])


        # ----- SPACE ----- # 
        self.action_space = [spaces.Discrete(3), spaces.Discrete(3)] # 0=Left, 1=Right, 2=Forward
        
        self.observation_space = [
            # So like these are the five things I want to measure in the observation space:
                # - Trapped=-1.0 to Not Trapped=1.0
                # - Distance from Opponent HEAD X direction from -1.0 to 1.0 normalized
                # - Distance from Opponent HEAD Y direction from -1.0 to 1.0 normalized
                # - Dot product between you and opponent direction vectors, so like parallel = [-1.0,0)U(0,-1.0) or perp=0
                # ^ Normalized
            spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        ]

        self.observation_state = [np.zeros(4), np.zeros(4)]
        self.steps_taken = 0
        self.max_moves = 300

        pass

    def reset(self, seed=None, options=None):
        # ----------
        # Returning the observation of the initial state
        # Reset the environment to the initial state so new episode can be run
        # ----------

        self.state = [np.random.uniform(-1.0, 1.0, (4,)), np.random.uniform(-1.0, 1.0, (4,))]
        self.trails = [set(), set()]
        self.agent_positions = [(random.randint(0, self.board_width), random.randint(0, self.board_height)),
                                (random.randint(0, self.board_width), random.randint(0, self.board_height))]
        
        self.trail[0].add(self.agent_positions[0])
        self.trail[1].add(self.agent_positions[1])

        self.steps_taken = 0
        
        return self.state, {}
    
    def collision(self, action):
        pass
    def step(self, actions):
        # ----------
        # Computes the NEXT OBSERVATION, the reward, and optional info like what's going in the environment
        # This is basically how the game is played and functionality
        # Both agents need to play at the same time
        # ----------
        self.steps_taken += 1 # A move has been played no matter if you went forward left or right
        done = self.steps_taken >= self.max_moves

        # Sampling moves
        agent1_action, agent2_action = actions[0], actions[1]


        
        # Update state 
        self.state[0] = agent1_next_move
        self.state[1] = agent2_next_move

        # Rewards
        agent1_reward = 1.0 if actions[0] == 0 else -1.0
        agent2_reward = 1.0 if actions[1] == 0 else -1.0
        rewards = [agent1_reward, agent2_reward]

        steps_dict = {"Total Moves": self.steps_taken}

        return self.state, rewards, [done, done], [False, False], steps_dict

    def render(self, mode=metadata["render.modes"]):
        # ----------
        # Returns Nothing but show the current environment
        # ----------
        print(f"Agent 1 State: {self.state[0]}, Agent 2 State: {self.state[1]}, Total Moves: {self.steps_taken}")


    def close(self):
        # ----------
        # Used to just cleanup resources, but like fuck this lol I don't got the energy to finish this method bro
        # ----------
        pass

    # WE DON'T NEED STOCHASTIC BEHAVIOUR


