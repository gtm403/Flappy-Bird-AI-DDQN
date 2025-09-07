import gymnasium
import flappy_bird_gymnasium
import numpy as np
from agent import DDQNAgent
import matplotlib.pyplot as plt
from collections import deque
import time
import os
from datetime import datetime

def train_ddqn(
    episodes=2000,
    max_steps_per_episode=10000,
    use_lidar=False,
    render_mode=None,
    save_interval=100,
    plot_interval=50
):

    # Create environment
    env = gymnasium.make("FlappyBird-v0", render_mode=render_mode, use_lidar=use_lidar)
    
    # Get state and action sizes
    state_size = 180 if use_lidar else 12
    action_size = env.action_space.n
    
    # Hyperparameters
    hyperparams = {
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': 10000,
        'batch_size': 32,
        'target_update_freq': 1000
    }
    
    # Create agent
    agent = DDQNAgent(state_size, action_size, **hyperparams)
    
    # Training metrics
    scores = deque(maxlen=100)
    all_scores = []
    all_losses = []
    best_score = -float('inf')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/ddqn_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save hyperparameters
    with open(f"{model_dir}/hyperparams.txt", 'w') as f:
        f.write(f"Episodes: {episodes}\n")
        f.write(f"Use LIDAR: {use_lidar}\n")
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Training DDQN on Flappy Bird ({'LIDAR' if use_lidar else 'Standard'} observations)")
    print(f"Device: {agent.device}")
    print("-" * 50)
    
    start_time = time.time()
    
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0
        steps = 0
        episode_losses = []
        
        for step in range(max_steps_per_episode):
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            done = terminated or truncated
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss > 0:
                    episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(total_reward)
        all_scores.append(total_reward)
        if episode_losses:
            all_losses.append(np.mean(episode_losses))
        
        if total_reward > best_score:
            best_score = total_reward
            agent.save(f"{model_dir}/best_model.pt")
        
        if episode % save_interval == 0:
            agent.save(f"{model_dir}/model_episode_{episode}.pt")
        
        if episode % 10 == 0:
            avg_score = np.mean(scores)
            elapsed_time = time.time() - start_time
            print(f"Episode: {episode:4d} | Avg Score: {avg_score:7.2f} | "
                  f"Last Score: {total_reward:7.2f} | Best: {best_score:7.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | Steps: {steps:4d} | "
                  f"Time: {elapsed_time/60:.1f}m")
        
        if episode % plot_interval == 0 and episode > 0:
            plot_training_progress(all_scores, all_losses, model_dir, episode)
    
    # Save final model
    agent.save(f"{model_dir}/final_model.pt")
    
    # Final plot
    plot_training_progress(all_scores, all_losses, model_dir, episodes)
    
    env.close()
    
    print("\nTraining completed!")
    print(f"Best score: {best_score}")
    print(f"Final average score (last 100 episodes): {np.mean(scores)}")
    print(f"Models saved in: {model_dir}")
    
    return agent, all_scores

def plot_training_progress(scores, losses, save_dir, episode):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(scores, alpha=0.3, color='blue', label='Score')
    if len(scores) > 100:
        # Moving average
        window = 100
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(scores)), moving_avg, color='red', 
                linewidth=2, label=f'{window}-episode moving average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Training Progress - Episode {episode}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if losses:
        ax2.plot(losses, alpha=0.5, color='green')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_progress_ep{episode}.png")
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DDQN agent on Flappy Bird')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--use-lidar', action='store_true', help='Use LIDAR observations')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--save-interval', type=int, default=100, help='Save model every N episodes')
    parser.add_argument('--plot-interval', type=int, default=50, help='Plot progress every N episodes')
    
    args = parser.parse_args()
    
    render_mode = "human" if args.render else None
    
    agent, scores = train_ddqn(
        episodes=args.episodes,
        use_lidar=args.use_lidar,
        render_mode=render_mode,
        save_interval=args.save_interval,
        plot_interval=args.plot_interval
    )
