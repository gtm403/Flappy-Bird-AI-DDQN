import gymnasium
import flappy_bird_gymnasium
import numpy as np
from agent import DDQNAgent
import matplotlib.pyplot as plt
from collections import deque
import time
import os
from datetime import datetime
import torch

def train_ddqn_optimized(
    episodes=5000,
    max_steps_per_episode=10000,
    use_lidar=False,
    render_mode=None,
    save_interval=100,
    plot_interval=50,
    checkpoint_path=None
):
    """Optimized DDQN training for Flappy Bird"""
    
    # Create environment
    env = gymnasium.make("FlappyBird-v0", render_mode=render_mode, use_lidar=use_lidar)
    
    # Get state and action sizes
    state_size = 180 if use_lidar else 12
    action_size = env.action_space.n
    
    # OPTIMIZED HYPERPARAMETERS
    hyperparams = {
        'learning_rate': 0.0001,        # Slightly lower for stability
        'gamma': 0.99,                  # Keep high for long-term planning
        'epsilon': 0.1 if checkpoint_path else 1.0,  # Lower starting epsilon if continuing
        'epsilon_min': 0.001,           # Lower minimum for fine-tuning
        'epsilon_decay': 0.9995,        # Slower decay for more exploration
        'memory_size': 50000,           # Much larger replay buffer
        'batch_size': 64,               # Larger batch for stability
        'target_update_freq': 500       # More frequent target updates
    }
    
    agent = DDQNAgent(state_size, action_size, **hyperparams)
    
    start_episode = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        agent.load(checkpoint_path)
        # Extract episode number if in filename
        if "episode_" in checkpoint_path:
            try:
                start_episode = int(checkpoint_path.split("episode_")[-1].split(".")[0]) + 1
            except:
                pass
    
    scores = deque(maxlen=100)
    pipes_history = deque(maxlen=100)
    all_scores = []
    all_pipes = []
    all_losses = []
    best_score = -float('inf')
    best_pipes = 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/ddqn_optimized_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    with open(f"{model_dir}/hyperparams.txt", 'w') as f:
        f.write(f"Episodes: {episodes}\n")
        f.write(f"Use LIDAR: {use_lidar}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Optimized DDQN Training ({'LIDAR' if use_lidar else 'Standard'} observations)")
    print(f"Device: {agent.device}")
    print("-" * 70)
    
    start_time = time.time()
    
    for episode in range(start_episode, start_episode + episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0
        steps = 0
        pipes_passed = 0
        episode_losses = []
        
        for step in range(max_steps_per_episode):
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            done = terminated or truncated
            
            # Count pipes
            if reward > 0.5:  # Pipe passed
                pipes_passed += 1
            
            # REWARD SHAPING (optional but helpful)
            shaped_reward = reward
            if use_lidar:
                # For LIDAR, add small penalty for getting too close to pipes
                min_distance = np.min(state)  # Closest LIDAR reading
                if min_distance < 20:  # Too close
                    shaped_reward -= 0.01
            
            # Store experience with shaped reward
            agent.remember(state, action, shaped_reward, next_state, done)
            
            # Train agent with more frequent updates
            if len(agent.memory) > agent.batch_size and step % 4 == 0:  # Train every 4 steps
                loss = agent.replay()
                if loss > 0:
                    episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(total_reward)
        pipes_history.append(pipes_passed)
        all_scores.append(total_reward)
        all_pipes.append(pipes_passed)
        if episode_losses:
            all_losses.append(np.mean(episode_losses))
        
        if pipes_passed > best_pipes or (pipes_passed == best_pipes and total_reward > best_score):
            best_pipes = pipes_passed
            best_score = total_reward
            agent.save(f"{model_dir}/best_model.pt")
            with open(f"{model_dir}/best_performance.txt", 'w') as f:
                f.write(f"Episode: {episode}\n")
                f.write(f"Pipes: {pipes_passed}\n")
                f.write(f"Score: {total_reward}\n")
        
        if episode % save_interval == 0:
            agent.save(f"{model_dir}/model_episode_{episode}.pt")
        
        if episode % 10 == 0:
            avg_score = np.mean(scores)
            avg_pipes = np.mean(pipes_history)
            elapsed_time = time.time() - start_time
            print(f"Episode: {episode:4d} | Avg Score: {avg_score:7.2f} | Avg Pipes: {avg_pipes:5.2f} | "
                  f"Last: {pipes_passed:2d} pipes | Best: {best_pipes:2d} pipes | "
                  f"ε: {agent.epsilon:.4f} | Time: {elapsed_time/60:.1f}m")
        
        # Plot progress
        if episode % plot_interval == 0 and episode > start_episode:
            plot_training_progress_optimized(all_scores, all_pipes, all_losses, model_dir, episode)
    
    agent.save(f"{model_dir}/final_model.pt")
    
    plot_training_progress_optimized(all_scores, all_pipes, all_losses, model_dir, start_episode + episodes - 1)
    
    env.close()
    
    print("\nTraining completed!")
    print(f"Best performance: {best_pipes} pipes (score: {best_score:.2f})")
    print(f"Final average: {np.mean(pipes_history):.2f} pipes")
    print(f"Models saved in: {model_dir}")
    
    return agent, all_scores, all_pipes

def plot_training_progress_optimized(scores, pipes, losses, save_dir, episode):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot scores
    ax1 = axes[0]
    ax1.plot(scores, alpha=0.3, color='blue', label='Score')
    if len(scores) > 100:
        window = 100
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(scores)), moving_avg, color='red', 
                linewidth=2, label=f'{window}-episode moving average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Training Progress - Episode {episode}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(pipes, alpha=0.3, color='green', label='Pipes passed')
    if len(pipes) > 100:
        window = 100
        moving_avg_pipes = np.convolve(pipes, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(pipes)), moving_avg_pipes, color='darkgreen', 
                linewidth=2, label=f'{window}-episode moving average')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Pipes Passed')
    ax2.set_title('Pipes Passed per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    if losses:
        ax3.plot(losses, alpha=0.5, color='orange')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_progress_ep{episode}.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized DDQN training for Flappy Bird')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--use-lidar', action='store_true', help='Use LIDAR observations')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--save-interval', type=int, default=100, help='Save model every N episodes')
    parser.add_argument('--plot-interval', type=int, default=50, help='Plot progress every N episodes')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint model to continue training')
    parser.add_argument('--no-lidar', dest='use_lidar', action='store_false', 
                       help='Use standard observations (default)')
    
    args = parser.parse_args()
    
    render_mode = "human" if args.render else None
    
    agent, scores, pipes = train_ddqn_optimized(
        episodes=args.episodes,
        use_lidar=args.use_lidar,
        render_mode=render_mode,
        save_interval=args.save_interval,
        plot_interval=args.plot_interval,
        checkpoint_path=args.checkpoint
    )
