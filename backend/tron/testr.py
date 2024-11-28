import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np

class TronBaseEnvTwoPlayer(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, board_width=1024, board_height=768, speed=5):
        super(TronBaseEnvTwoPlayer, self).__init__()
        
        # ----- BOUNDS ----- # 
        self.board_width = board_width
        self.board_height = board_height
        
        # ----- AGENTS ----- #
        self.speed = speed # Reduced speed for more controlled movement
        
        # More varied initial directions
        self.direction = [45, 225]  # Diagonal directions

        # More spread out initial positions
        self.agent_positions = [
            (int(self.board_width * 0.2), int(self.board_height * 0.5)),
            (int(self.board_width * 0.8), int(self.board_height * 0.5))
        ]

        # Instantiating the starting positions of the trails and the sets that will hold each agents trails
        self.trails = [set(), set()]
        self.trails[0].add(self.agent_positions[0])
        self.trails[1].add(self.agent_positions[1])

        # ----- SPACE ----- # 
        self.action_space = [spaces.Discrete(3), spaces.Discrete(3)] # 0=Left, 1=Right, 2=Forward
        
        self.observation_space = [
            spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        ]

        self.state = [np.zeros(3), np.zeros(3)]
        self.steps_taken = 0
        self.max_moves = 300
        self.agent1_reward = 0
        self.agent2_reward = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.agent1_reward = 0
        self.agent2_reward = 0

        # Randomize initial state while keeping agents apart
        self.direction = [
            random.randint(0, 360),  # Random initial direction for agent 1
            random.randint(0, 360)   # Random initial direction for agent 2
        ]
        
        self.agent_positions = [
            (int(self.board_width * 0.2), int(self.board_height * 0.5)),
            (int(self.board_width * 0.8), int(self.board_height * 0.5))
        ]
        
        self.trails = [set(), set()]
        self.trails[0].add(self.agent_positions[0])
        self.trails[1].add(self.agent_positions[1])
        
        self.steps_taken = 0
        
        # Compute initial state
        x_diff_agent1 = (self.agent_positions[1][0] - self.agent_positions[0][0]) / self.board_width
        y_diff_agent1 = (self.agent_positions[1][1] - self.agent_positions[0][1]) / self.board_height
        
        x_diff_agent2 = (self.agent_positions[0][0] - self.agent_positions[1][0]) / self.board_width
        y_diff_agent2 = (self.agent_positions[0][1] - self.agent_positions[1][1]) / self.board_height
        
        # Compute dot product of initial directions
        agent1_dir_vec = np.array([np.cos(np.radians(self.direction[0])), np.sin(np.radians(self.direction[0]))])
        agent2_dir_vec = np.array([np.cos(np.radians(self.direction[1])), np.sin(np.radians(self.direction[1]))])
        dot_product = np.clip(np.dot(agent1_dir_vec, agent2_dir_vec), -1.0, 1.0)
        
        self.state = [
            np.array([x_diff_agent1, y_diff_agent1, dot_product]),
            np.array([x_diff_agent2, y_diff_agent2, dot_product])
        ]
        
        return self.state, {}
    
    def _compute_new_position(self, current_pos, action, agent_id):
        x, y = current_pos
        direction = self.direction[agent_id]  
        
        # Debug print
        print(f"Agent {agent_id}: Initial pos={current_pos}, direction={direction}, action={action}")

        # Turn left or right first
        if action == 0:  # Left
            direction -= 90  
        elif action == 1:  # Right
            direction += 90  
        
        # Ensure the direction stays within the 0-360 degree range
        direction = direction % 360

        # Then move forward if action is 2
        if action == 2:
            x += int(self.speed * np.cos(np.radians(direction)))
            y += int(self.speed * np.sin(np.radians(direction)))

        # Update the agent's direction after the move
        self.direction[agent_id] = direction
        
        print(f"Agent {agent_id}: New pos={x, y}, new direction={direction}")
        return (x, y), direction
    
    def _is_collision(self, position, agent_id):
        x, y = position
        # Expanded out-of-bounds check with a small buffer
        if (x < 0 or x >= self.board_width or 
            y < 0 or y >= self.board_height):
            print(f"Agent {agent_id}: Out of bounds at {position}")
            return True

        # Check trail collisions with more verbose output
        if agent_id == 0:
            if position in self.trails[0]:
                print(f"Agent 0: Hit own trail at {position}")
                return True
            if position in self.trails[1]:
                print(f"Agent 0: Hit opponent's trail at {position}")
                return True
        
        elif agent_id == 1:
            if position in self.trails[1]:
                print(f"Agent 1: Hit own trail at {position}")
                return True
            if position in self.trails[0]:
                print(f"Agent 1: Hit opponent's trail at {position}")
                return True
        
        return False
            
    def step(self, actions):
        self.steps_taken += 1
        done = False
        
        print(f"\n--- Step {self.steps_taken} ---")
        agent1_action, agent2_action = actions[0], actions[1]

        new_position_agent1, agent1_direction = self._compute_new_position(self.agent_positions[0], agent1_action, 0)
        new_position_agent2, agent2_direction = self._compute_new_position(self.agent_positions[1], agent2_action, 1)

        # Check if either agent is out of bounds or hits a trail
        agent1_collision = self._is_collision(new_position_agent1, 0)
        agent2_collision = self._is_collision(new_position_agent2, 1)

        # Reward and termination logic
        if agent1_collision and agent2_collision:
            print("Both agents collided simultaneously!")
            agent1_reward = -400.0
            agent2_reward = -400.0
            done = True
        elif agent1_collision:
            print("Agent 1 collided!")
            agent1_reward = -400.0
            agent2_reward = 400.0
            done = True
        elif agent2_collision:
            print("Agent 2 collided!")
            agent2_reward = -400.0
            agent1_reward = 400.0
            done = True
        else:
            # No collision, continue game
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
                print("Max moves reached!")
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

        print(f"Agent 1 position: {self.agent_positions[0]}")
        print(f"Agent 2 position: {self.agent_positions[1]}")

        return self.state, rewards, [done], [False, False], steps_dict

    def render(self, mode=metadata["render.modes"]):
        print(f"Agent 1 State: {self.state[0]}, Agent 2 State: {self.state[1]}, Total Moves: {self.steps_taken}")

    def close(self):
        pass