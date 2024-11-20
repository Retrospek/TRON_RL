from entire import Tron2Env  # Make sure this import matches your file structure

def test_environment():
    env = Tron2Env()
    obs = env.reset()
    
    # Run for a few steps
    for _ in range(10):  # Test 10 steps
        # Example actions: Agent 1 moves right (1), Agent 2 moves left (3)
        actions = [1, 3]
        
        obs, rewards, dones, _, info = env.step(actions)
        env.render()  # This will show the game window
        
        print(f"Step {info['steps_taken']}:")
        print(f"Agent 1 Position: {obs['agent1']}")
        print(f"Agent 2 Position: {obs['agent2']}")
        print(f"Rewards: {rewards}")
        print(f"Done: {dones}")
        print("-" * 50)
        
        if dones[0]:
            break

if __name__ == "__main__":
    test_environment()