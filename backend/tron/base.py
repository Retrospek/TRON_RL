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
            (int(self.board_width / 4), int(self.board_height / 2)),
            (int(self.board_width - self.board_width / 4), int(self.board_height / 2))
        ]

        # Instantiating the starting positions of the trails and the sets that will hold each agents trails
        self.trails = [set(), set()]
        self.trails[0].add(self.agent_positions[0])
        self.trails[1].add(self.agent_positions[1])


        # ----- SPACE ----- # 
        self.action_space = [spaces.Discrete(3), spaces.Discrete(3)] # 0=Left, 1=Right, 2=Forward
        
        self.observation_space = [
            # So like these are the five things I want to measure in the observation space:
                # - Distance from Opponent HEAD X direction from -1.0 to 1.0 normalized
                # - Distance from Opponent HEAD Y direction from -1.0 to 1.0 normalized
                # - Dot product between you and opponent direction vectors, so like parallel = [-1.0,0)U(0,-1.0) or perp=0
                # ^ Normalized
            spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        ]

        self.state = [np.zeros(3), np.zeros(3)]
        self.steps_taken = 0
        self.max_moves = 300
        self.agent1_reward = 0
        self.agent2_reward = 0

        pass

    def reset(self, seed=None, options=None):
        # ----------
        # Returning the observation of the initial state
        # Reset the environment to the initial state so new episode can be run
        # ----------
        self.agent1_reward = 0
        self.agent2_reward = 0

        self.state = [np.random.uniform(-1.0, 1.0, (3,)), np.random.uniform(-1.0, 1.0, (3,))]
        self.trails = [set(), set()]
        self.agent_positions = [
            (int(self.board_width / 4), int(self.board_height / 2)),
            (int(3 * self.board_width / 4), int(self.board_height / 2))
        ]

        
        self.trails[0].add(self.agent_positions[0])
        self.trails[1].add(self.agent_positions[1])
        
        self.steps_taken = 0
        
        return self.state, {}
    
    def _compute_new_position(self, current_pos, action, agent_id):
        # ----- Logic ----- #
        # Compute the new position of the agent based on the action taken
        # If the action is 0, move left, if 1, move right, if not either which is 2 then MOVE
        # Actions involving turn don't move the the thing they just turn the direction angle
        # Agent id = 0 is Agent 1
        # Agend id = 1 is Agent 2

        same_loc = True
        x, y = current_pos
        direction = self.direction[agent_id]  
        #print(direction)
        if action == 0:  # Left
            direction -= 90  
        elif action == 1:  # Right
            direction += 90  
        elif action == 2:  # Forward makes the thing actually move not the turns themselves
            x += int(self.speed * np.cos(np.radians(int(direction)))) # if deg = 0 then cos = 1 and if deg = 90 then cos = 0, so yeah
            y += int(self.speed * np.sin(np.radians(int(direction))))  # Move based on current direction
            same_loc = False
        # Ensure the direction stays within the 0-360 degree range
        direction = direction % 360 # Need this after several moves made

        # Update the agent's direction after the move
        self.direction[agent_id] = direction
        
        return (x, y), direction, same_loc
    
    def _is_collision(self, position, agent_id, same_loc):
        # ----- Logic ----- #
        # Check if the agent has collided with the wall or the other agent
        # But before doing that check if the agent has even moved because then same_loc would be False
        # If same_loc == True then that means it turned right because in the step method where it computes new position
        # the position stays the same when you turn other wise you move a certain self.speed forward
        #print(f"same_loc bool: {same_loc}")
        x, y = position
        if same_loc == False:
            if agent_id == 0: # This shit for first Agent
                if not (0 <= x < self.board_width and 0 <= y < self.board_height):
                    return True  # Out of bounds
                if position in self.trails[0] or position in self.trails[1]: # Same HERE
                    return True # If hitting trail

            elif agent_id == 1:
                if not (0 <= x < self.board_width and 0 <= y < self.board_height):
                    return True  # Out of bounds
                if position in self.trails[0] or position in self.trails[1]:
                    return True # If hitting trail
            return False
        
        else: # If just turned and didn't move
            return False

            
    def step(self, actions):
        # ----------
        # Computes the NEXT OBSERVATION, the reward, and optional info like what's going in the environment
        # This is basically how the game is played and functionality
        # Both agents need to play at the same time
        # ----------
    
        self.steps_taken += 1 # A move has been played no matter if you went forward left or right
        done = False  # Start with done as False
        
        agent1_action, agent2_action = actions[0], actions[1]

        new_position_agent1, agent1_direction, same_loc_agent1 = self._compute_new_position(self.agent_positions[0], agent1_action, 0)
        new_position_agent2, agent2_direction, same_loc_agent2 = self._compute_new_position(self.agent_positions[1], agent2_action, 1)

        # Check if either agent is out of bounds or hits a trail
        agent1_collision = self._is_collision(new_position_agent1, 0, same_loc_agent1)
        agent2_collision = self._is_collision(new_position_agent2, 1, same_loc_agent2)

        # Reward and termination logic
        if agent1_collision and agent2_collision:
            # Both agents collide simultaneously
            agent1_reward = -400.0
            agent2_reward = -400.0
            done = True
        elif agent1_collision:
            # Agent 1 collides
            agent1_reward = -400.0
            agent2_reward = 400.0
            done = True
        elif agent2_collision:
            # Agent 2 collides
            agent2_reward = -400.0
            agent1_reward = 400.0
            done = True
        else:
            # No collision, continue game
            # Update agent positions
            self.agent_positions[0] = new_position_agent1
            self.agent_positions[1] = new_position_agent2
            
            # Add new positions to trails
            self.trails[0].add(self.agent_positions[0]) 
            self.trails[1].add(self.agent_positions[1])  

            # Increment rewards for survival
            self.agent1_reward += 1
            self.agent2_reward += 1
            
            agent1_reward = self.agent1_reward
            agent2_reward = self.agent2_reward
            
            # Check for max moves
            if self.steps_taken >= self.max_moves:
                done = True

        # Compute state (relative positions and dot product)
        x_diff_agent1 = (self.agent_positions[1][0] - self.agent_positions[0][0]) / self.board_width
        y_diff_agent1 = (self.agent_positions[1][1] - self.agent_positions[0][1]) / self.board_height
        x_diff_agent2 = (self.agent_positions[0][0] - self.agent_positions[1][0]) / self.board_width
        y_diff_agent2 = (self.agent_positions[0][1] - self.agent_positions[1][1]) / self.board_height

        # Convert directions to unit vectors for dot product
        agent1_dir_vec = np.array([np.cos(np.radians(agent1_direction)), np.sin(np.radians(agent1_direction))])
        agent2_dir_vec = np.array([np.cos(np.radians(agent2_direction)), np.sin(np.radians(agent2_direction))])
        dot_product = np.clip(np.dot(agent1_dir_vec, agent2_dir_vec), -1.0, 1.0)

        # Update state
        self.state[0] = np.array([x_diff_agent1, y_diff_agent1, dot_product])
        self.state[1] = np.array([x_diff_agent2, y_diff_agent2, dot_product])

        # Prepare return values
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


