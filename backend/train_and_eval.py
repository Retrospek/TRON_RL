import torch
import numpy as np
from tron_env import Tron2Env
import matplotlib.pyplot as plt
from collections import deque
import time
import os
from datetime import datetime

def create_training_directory():
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    directory = f'training_results_{timestamp}'
    os.makedirs(directory, exist_ok=True)
    return directory

def plot_metrics(metrics, title, ylabel, save_path):
    
    plt.figure(figsize=(10, 6))
    plt.plot(metrics)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def train_agents(episodes=1000, max_steps=500, render_interval=100, save_interval=100):
    """Train two DQN agents against each other"""
    # Create directory for saving results
    save_dir = create_training_directory()
    
    # Initialize environment and agents
    env = Tron2Env()
    
    # Initialize agents with different exploration rates
    agent1 = DQNAgent(state_dim=1, action_dim=4, agent_id=1)
    agent2 = DQNAgent(state_dim=1, action_dim=4, agent_id=2)
    
    # Metrics tracking
    episode_rewards = []
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    win_rates = []
    avg_episode_lengths = []
    losses = {'agent1': [], 'agent2': []}
    rolling_reward_window = 100
    rolling_rewards = deque(maxlen=rolling_reward_window)
    
    print("Starting training...")
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = {'agent1': [], 'agent2': []}
        
        for step in range(max_steps):
            # Get actions from both agents
            action1 = agent1.select_action(state)
            action2 = agent2.select_action(state)
            
            # Take step in environment
            next_state, rewards, dones, _, info = env.step([action1, action2])
            
            # Store experiences in replay buffer
            agent1.memory.push(state, action1, rewards[0], next_state, dones[0])
            agent2.memory.push(state, action2, rewards[1], next_state, dones[1])
            
            # Train both agents
            if len(agent1.memory) > agent1.batch_size:
                loss1 = agent1.train()
                loss2 = agent2.train()
                episode_losses['agent1'].append(loss1)
                episode_losses['agent2'].append(loss2)
            
            episode_reward += sum(rewards)
            episode_length += 1
            
            # Render every render_interval episodes
            if episode % render_interval == 0:
                env.render()
            
            # Move to next state
            state = next_state
            
            # Match Outcomes stored here
            if dones[0]:
                
                if rewards[0] > rewards[1]:
                    agent1_wins += 1
                elif rewards[1] > rewards[0]:
                    agent2_wins += 1
                else:
                    draws += 1
                break
        
        # Record metrics
        episode_rewards.append(episode_reward)
        rolling_rewards.append(episode_reward)
        avg_episode_lengths.append(episode_length)
        
        if episode_losses['agent1']:
            losses['agent1'].append(np.mean(episode_losses['agent1']))
            losses['agent2'].append(np.mean(episode_losses['agent2']))
        
        win_rates.append(agent1_wins / (episode + 1))
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(list(rolling_rewards))
            time_elapsed = time.time() - start_time
            print(f"\nEpisode {episode + 1}/{episodes} (Time elapsed: {time_elapsed:.1f}s)")
            print(f"Average Reward (last {rolling_reward_window}): {avg_reward:.2f}")
            print(f"Episode Length: {episode_length}")
            print(f"Agent 1 - Wins: {agent1_wins}, Win Rate: {agent1_wins/(episode+1):.2%}")
            print(f"Agent 2 - Wins: {agent2_wins}, Win Rate: {agent2_wins/(episode+1):.2%}")
            print(f"Draws: {draws}")
            print(f"Epsilon 1: {agent1.epsilon:.3f}, Epsilon 2: {agent2.epsilon:.3f}")
            print("-" * 50)
        
        # Save models periodically
        if (episode + 1) % save_interval == 0:
            agent1.save(f"{save_dir}/agent1_episode_{episode+1}.pth")
            agent2.save(f"{save_dir}/agent2_episode_{episode+1}.pth")
    
    # Save final models
    agent1.save(f"{save_dir}/agent1_final.pth")
    agent2.save(f"{save_dir}/agent2_final.pth")
    
    # Plot and save metrics
    plot_metrics(episode_rewards, 'Episode Rewards', 'Reward', f"{save_dir}/episode_rewards.png")
    plot_metrics(win_rates, 'Agent 1 Win Rate', 'Win Rate', f"{save_dir}/win_rates.png")
    plot_metrics(avg_episode_lengths, 'Average Episode Length', 'Steps', f"{save_dir}/episode_lengths.png")
    plot_metrics(losses['agent1'], 'Agent 1 Training Loss', 'Loss', f"{save_dir}/agent1_loss.png")
    plot_metrics(losses['agent2'], 'Agent 2 Training Loss', 'Loss', f"{save_dir}/agent2_loss.png")
    
    # Save final statistics to a file
    with open(f"{save_dir}/training_stats.txt", 'w') as f:
        f.write(f"Total Episodes: {episodes}\n")
        f.write(f"Agent 1 Wins: {agent1_wins}\n")
        f.write(f"Agent 2 Wins: {agent2_wins}\n")
        f.write(f"Draws: {draws}\n")
        f.write(f"Agent 1 Final Win Rate: {agent1_wins/episodes:.2%}\n")
        f.write(f"Agent 2 Final Win Rate: {agent2_wins/episodes:.2%}\n")
        f.write(f"Draw Rate: {draws/episodes:.2%}\n")
        f.write(f"Total Training Time: {time.time() - start_time:.1f} seconds\n")
    
    return {
        'episode_rewards': episode_rewards,
        'win_rates': win_rates,
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'draws': draws,
        'losses': losses,
        'avg_episode_lengths': avg_episode_lengths,
        'save_dir': save_dir
    }

def evaluate_agents(agent1_path, agent2_path, num_games=100):
    """Evaluate two trained agents against each other"""
    env = Tron2Env()
    
    # Load trained agents
    agent1 = DQNAgent(state_dim=1, action_dim=4, agent_id=1)
    agent2 = DQNAgent(state_dim=1, action_dim=4, agent_id=2)
    agent1.load(agent1_path)
    agent2.load(agent2_path)
    
    # Set to evaluation mode (no exploration)
    agent1.epsilon = 0
    agent2.epsilon = 0
    
    wins = {'agent1': 0, 'agent2': 0, 'draws': 0}
    game_lengths = []
    
    for game in range(num_games):
        state = env.reset()
        game_length = 0
        
        while True:
            action1 = agent1.select_action(state)
            action2 = agent2.select_action(state)
            
            next_state, rewards, dones, _, _ = env.step([action1, action2])
            game_length += 1
            
            if dones[0]:
                if rewards[0] > rewards[1]:
                    wins['agent1'] += 1
                elif rewards[1] > rewards[0]:
                    wins['agent2'] += 1
                else:
                    wins['draws'] += 1
                game_lengths.append(game_length)
                break
            
            state = next_state
        
        if (game + 1) % 10 == 0:
            print(f"Evaluated {game + 1} games...")
    
    print("\nEvaluation Results:")
    print(f"Agent 1 Wins: {wins['agent1']} ({wins['agent1']/num_games:.2%})")
    print(f"Agent 2 Wins: {wins['agent2']} ({wins['agent2']/num_games:.2%})")
    print(f"Draws: {wins['draws']} ({wins['draws']/num_games:.2%})")
    print(f"Average Game Length: {np.mean(game_lengths):.1f} steps")
    
    return wins, game_lengths

if __name__ == "__main__":
    # Train the agents
    print("Starting training phase...")
    results = train_agents(episodes=1000, max_steps=500)
    
    # Evaluate the final models
    print("\nStarting evaluation phase...")
    final_agent1_path = f"{results['save_dir']}/agent1_final.pth"
    final_agent2_path = f"{results['save_dir']}/agent2_final.pth"
    evaluation_results, game_lengths = evaluate_agents(final_agent1_path, final_agent2_path, num_games=100)
    
    # Save evaluation results
    with open(f"{results['save_dir']}/evaluation_results.txt", 'w') as f:
        f.write("Evaluation Results:\n")
        f.write(f"Number of evaluation games: 100\n")
        f.write(f"Agent 1 Wins: {evaluation_results['agent1']}\n")
        f.write(f"Agent 2 Wins: {evaluation_results['agent2']}\n")
        f.write(f"Draws: {evaluation_results['draws']}\n")
        f.write(f"Average Game Length: {np.mean(game_lengths):.1f} steps\n")