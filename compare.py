import gymnasium
import flappy_bird_gymnasium
import numpy as np
import matplotlib.pyplot as plt
from agent import DDQNAgent
from train import train_ddqn
import os

def compare_observation_types(episodes_per_type=1000):
    
    print("Comparing LIDAR vs Standard observations")
    print("=" * 60)
    
    results = {}
    
    # Train with standard observations
    print("\n1. Training with STANDARD observations (12 features)")
    print("-" * 60)
    agent_std, scores_std = train_ddqn(
        episodes=episodes_per_type,
        use_lidar=False,
        render_mode=None,
        save_interval=200,
        plot_interval=100
    )
    results['standard'] = {
        'agent': agent_std,
        'scores': scores_std,
        'final_avg': np.mean(scores_std[-100:]),
        'best': np.max(scores_std)
    }
    
    # Train with LIDAR observations
    print("\n2. Training with LIDAR observations (180 features)")
    print("-" * 60)
    agent_lidar, scores_lidar = train_ddqn(
        episodes=episodes_per_type,
        use_lidar=True,
        render_mode=None,
        save_interval=200,
        plot_interval=100
    )
    results['lidar'] = {
        'agent': agent_lidar,
        'scores': scores_lidar,
        'final_avg': np.mean(scores_lidar[-100:]),
        'best': np.max(scores_lidar)
    }
    
    plot_comparison(results, episodes_per_type)
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Standard Observations:")
    print(f"  - Final Average Score: {results['standard']['final_avg']:.2f}")
    print(f"  - Best Score: {results['standard']['best']:.2f}")
    print(f"  - Convergence Speed: Episode {find_convergence(scores_std)}")
    
    print(f"\nLIDAR Observations:")
    print(f"  - Final Average Score: {results['lidar']['final_avg']:.2f}")
    print(f"  - Best Score: {results['lidar']['best']:.2f}")
    print(f"  - Convergence Speed: Episode {find_convergence(scores_lidar)}")
    
    return results

def find_convergence(scores, window=100, threshold=0.9):
    if len(scores) < window:
        return len(scores)
    
    final_avg = np.mean(scores[-window:])
    target = final_avg * threshold
    
    for i in range(window, len(scores)):
        if np.mean(scores[i-window:i]) >= target:
            return i
    
    return len(scores)

def plot_comparison(results, episodes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    window = 50
    for obs_type, data in results.items():
        scores = data['scores']
        ax1.plot(scores, alpha=0.3, label=f'{obs_type.capitalize()} (raw)')
        
        if len(scores) > window:
            moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(scores)), moving_avg, linewidth=2, 
                    label=f'{obs_type.capitalize()} (avg)')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Training Progress Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final performance box plot
    test_episodes = 100
    test_scores = {'Standard': [], 'LIDAR': []}
    
    print("\nTesting final performance...")
    for obs_type, label in [('standard', 'Standard'), ('lidar', 'LIDAR')]:
        agent = results[obs_type]['agent']
        agent.epsilon = 0  # No exploration
        
        env = gymnasium.make("FlappyBird-v0", use_lidar=(obs_type=='lidar'))
        
        for _ in range(test_episodes):
            state, _ = env.reset()
            total_reward = 0
            
            while True:
                action = agent.act(np.array(state, dtype=np.float32), training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            test_scores[label].append(total_reward)
        
        env.close()
    
    ax2.boxplot([test_scores['Standard'], test_scores['LIDAR']], 
                labels=['Standard', 'LIDAR'])
    ax2.set_ylabel('Score')
    ax2.set_title(f'Final Performance ({test_episodes} test episodes)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('observation_comparison.png')
    plt.show()

def analyze_observation_space():
    """Analyze and visualize the observation spaces"""
    print("\nAnalyzing Observation Spaces")
    print("=" * 60)
    
    # Standard observations
    env_std = gymnasium.make("FlappyBird-v0", use_lidar=False)
    state_std, _ = env_std.reset()
    
    print("Standard Observations (12 features):")
    print("1. Last pipe horizontal position")
    print("2. Last top pipe vertical position")
    print("3. Last bottom pipe vertical position")
    print("4. Next pipe horizontal position")
    print("5. Next top pipe vertical position")
    print("6. Next bottom pipe vertical position")
    print("7. Next next pipe horizontal position")
    print("8. Next next top pipe vertical position")
    print("9. Next next bottom pipe vertical position")
    print("10. Player vertical position")
    print("11. Player vertical velocity")
    print("12. Player rotation")
    print(f"\nExample state: {np.array(state_std).round(2)}")
    
    env_std.close()
    

    env_lidar = gymnasium.make("FlappyBird-v0", use_lidar=True)
    state_lidar, _ = env_lidar.reset()
    
    print("\n\nLIDAR Observations (180 features):")
    print("- 180 distance readings in a semicircle")
    print("- Each reading represents distance to nearest obstacle")
    print(f"- Shape: {np.array(state_lidar).shape}")
    print(f"- Min value: {np.min(state_lidar):.2f}")
    print(f"- Max value: {np.max(state_lidar):.2f}")
    print(f"- Mean value: {np.mean(state_lidar):.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(state_lidar)
    plt.title('LIDAR Distance Readings')
    plt.xlabel('Angle Index (0-179)')
    plt.ylabel('Distance')
    plt.grid(True, alpha=0.3)
    
    # Polar plot of LIDAR
    plt.subplot(2, 1, 2, projection='polar')
    angles = np.linspace(0, np.pi, 180)
    plt.plot(angles, state_lidar)
    plt.title('LIDAR Readings (Polar View)')
    
    plt.tight_layout()
    plt.savefig('lidar_visualization.png')
    plt.show()
    
    env_lidar.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare DDQN performance with different observations')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Episodes to train each agent type')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze observation spaces without training')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_observation_space()
    else:
        analyze_observation_space()
        compare_observation_types(args.episodes)
