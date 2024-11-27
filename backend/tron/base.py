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
    
    def _compute_new_position(self, current_pos, action):
        x, y = current_pos
        if action == 0:  # Left
            x -= self.speed
        elif action == 1:  # Right
            x += self.speed
        elif action == 2:  # Forward
            y -= self.speed
        return (x, y)
    
    def _is_collision(self, position, agent_id):
        if position in self.trails[0] or position in self.trails[1]:
            return True
        return False
            
    def step(self, actions):
        # ----------
        # Computes the NEXT OBSERVATION, the reward, and optional info like what's going in the environment
        # This is basically how the game is played and functionality
        # Both agents need to play at the same time
        # ----------
        self.steps_taken += 1 # A move has been played no matter if you went forward left or right
        done = self.steps_taken >= self.max_moves

        agent1_action, agent2_action = actions[0], actions[1]

        new_position_agent1 = self._compute_new_position(self.agent_positions[0], agent1_action)
        new_position_agent2 = self._compute_new_position(self.agent_positions[1], agent2_action)

        # Collision check for Agent 1
        if self._check_collision(new_position_agent1, 0):  # Check if Agent 1 collides
            agent1_reward = -10.0  # Damn you agent 1 lost
            agent2_reward = 10.0 # AMAZING!!!! Agent 1 won
            done = True  # End the game if there's a collision
        #else:
        #    agent1_reward = 1.0 if actions[0] == 0 else -1.0  # Normal reward

        # Collision check for Agent 2
        if self._check_collision(new_position_agent2, 1):  # Check if Agent 2 collides
            agent2_reward = -10.0  # Damn you agent 2 lost
            agent1_reward = 10.0 # AMAZING!!!! Agent 1 won
            done = True  # End the game if there's a collision

        #else:
        #    agent2_reward = 1.0 if actions[1] == 0 else -1.0  # Normal reward

        # Update agent positions if no collision
        if not done:
            self.agent_positions[0] = new_position_agent1
            self.agent_positions[1] = new_position_agent2
            self.trails[0].add(self.agent_positions[0]) 
            self.trails[1].add(self.agent_positions[1])  

        rewards = [agent1_reward, agent2_reward]

        steps_dict = {"Total Moves": self.steps_taken}

        return self.state, rewards, [done], [False, False], steps_dict

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


