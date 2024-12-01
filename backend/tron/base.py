import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TronBaseEnvTwoPlayer(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(TronBaseEnvTwoPlayer, self).__init__()

        # Board dimensions
        self.board_width = 1024
        self.board_height = 768
        
        # Agent properties
        self.speed = 15  # Movement speed
        self.direction = [0, 180]  # Initial directions (angles in degrees)

        self.agent_positions = [
            (int(self.board_width / 4), int(self.board_height / 2)),
            (int(3 * self.board_width / 4), int(self.board_height / 2))
        ]

        # Trails for collision detection
        self.trails = [set(), set()]
        self.trails[0].add(self.agent_positions[0])
        self.trails[1].add(self.agent_positions[1])

        # Action and observation spaces
        self.action_space = [spaces.Discrete(3), spaces.Discrete(3)]  # Actions: 0=Left, 1=Right, 2=Forward
        self.observation_space = [
            spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        ]

        # State variables
        self.state = [np.zeros(3), np.zeros(3)]
        self.steps_taken = 0
        self.max_moves = 500
        self.agent1_reward = 0
        self.agent2_reward = 0

    def reset(self, seed=None, options=None):
        self.agent1_reward = 0
        self.agent2_reward = 0
        self.steps_taken = 0

        self.state = [
            np.random.uniform(-1.0, 1.0, (3,)),
            np.random.uniform(-1.0, 1.0, (3,))
        ]
        self.trails = [set(), set()]
        self.agent_positions = [
            (int(self.board_width / 4), int(self.board_height / 2)),
            (int(3 * self.board_width / 4), int(self.board_height / 2))
        ]

        self.trails[0].add(self.agent_positions[0])
        self.trails[1].add(self.agent_positions[1])

        print(f"Reset: Agent 1 Position: {self.agent_positions[0]}, Agent 2 Position: {self.agent_positions[1]}")
        return self.state, {}

    def _compute_new_position(self, current_pos, action, agent_id):
        x, y = current_pos
        direction = self.direction[agent_id]
        change_direction = False
        if action == 0:  # Left
            direction -= 90
            change_direction = True
        elif action == 1:  # Right
            direction += 90
            change_direction = True
        elif action == 2:  # Forward
            x += int(self.speed * np.cos(np.radians(direction)))
            y += int(self.speed * np.sin(np.radians(direction)))

        direction = direction % 360  # Normalize direction
        self.direction[agent_id] = direction

        return (x, y), direction, change_direction

    def _is_collision(self, position, agent_id):
        x, y = position
        if not (0 <= x <= self.board_width and 0 <= y <= self.board_height):
            #print("Out of Board")
            return True  # Out of bounds
        #print(f" So first agent Trail: {self.trails[0]}")
        #print(f" So second agent Trail: {self.trails[1]}")
        if position in self.trails[0] or position in self.trails[1]:
            #print("Hitting Trail")
            return True  # Collision with trail
        return False

    def step(self, actions):
        self.steps_taken += 1
        done = False

        # Initialize rewards to 0 at the start of each step
        agent1_reward = 0
        agent2_reward = 0

        agent1_action, agent2_action = actions[0], actions[1]

        new_position_agent1, agent1_direction, chang_dir_agent1 = self._compute_new_position(self.agent_positions[0], agent1_action, 0)
        new_position_agent2, agent2_direction, chang_dir_agent2 = self._compute_new_position(self.agent_positions[1], agent2_action, 1)
        #print(f"Agent 1 Changed Direction: {chang_dir_agent1}")
        #print(f"Agent 2 Changed Direction: {chang_dir_agent2}")
        agent1_collision = self._is_collision(new_position_agent1, 0)
        agent2_collision = self._is_collision(new_position_agent2, 1)

        #print(f"""Agent 1 Collission Boolean: {agent1_collision}\nAgent 2 Collission Boolean: {agent2_collision}""")

        # Determine rewards and done state based on collision and direction change scenarios
        if not(chang_dir_agent1 and chang_dir_agent2):
            if agent1_collision and agent2_collision:
                agent1_reward = -400.0
                agent2_reward = -400.0
                done = True

        if not(chang_dir_agent1):
            if agent1_collision:
                agent1_reward = -400.0
                agent2_reward = 400.0
                done = True

        if not(chang_dir_agent2):
            if agent2_collision:
                agent1_reward = 400.0
                agent2_reward = -400.0
                done = True
        else:
            self.agent_positions[0] = new_position_agent1
            self.agent_positions[1] = new_position_agent2

            self.trails[0].add(new_position_agent1)
            self.trails[1].add(new_position_agent2)

            agent1_reward += 1
            agent2_reward += 1

            if self.steps_taken >= self.max_moves:
                done = True

        x_diff_agent1 = (self.agent_positions[1][0] - self.agent_positions[0][0]) / self.board_width
        y_diff_agent1 = (self.agent_positions[1][1] - self.agent_positions[0][1]) / self.board_height
        x_diff_agent2 = -x_diff_agent1
        y_diff_agent2 = -y_diff_agent1

        agent1_dir_vec = np.array([np.cos(np.radians(agent1_direction)), np.sin(np.radians(agent1_direction))])
        agent2_dir_vec = np.array([np.cos(np.radians(agent2_direction)), np.sin(np.radians(agent2_direction))])
        dot_product = np.clip(np.dot(agent1_dir_vec, agent2_dir_vec), -1.0, 1.0)

        self.state[0] = np.array([x_diff_agent1, y_diff_agent1, dot_product])
        self.state[1] = np.array([x_diff_agent2, y_diff_agent2, dot_product])

        rewards = [agent1_reward, agent2_reward]
        #print(f"Step {self.steps_taken}: Agent 1 Trail: {self.trails[0]}, Agent 2 Trail: {self.trails[1]}")

        return self.state, rewards, [done], [False, False], {"steps": self.steps_taken}

    def render(self, mode=metadata["render.modes"]):
        print(f"Render: Agent 1: {self.state[0]}, Agent 2: {self.state[1]}, Steps: {self.steps_taken}")

    def close(self):
        pass
