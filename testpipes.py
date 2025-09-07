import gymnasium
import flappy_bird_gymnasium
import numpy as np
from agent import DDQNAgent
import time
import argparse

def test_agent_with_pipe_count(model_path, episodes=10, use_lidar=False, render=True, delay=0.03):
    
    # Create environment
    render_mode = "human" if render else None
    env = gymnasium.make("FlappyBird-v0", render_mode=render_mode, use_lidar=use_lidar)
    
    state_size = 180 if use_lidar else 12
    action_size = env.action_space.n
    
    # Create and load agent
    agent = DDQNAgent(state_size, action_size)
    agent.load(model_path)
    agent.epsilon = 0  # No exploration during testing
    
    scores = []
    pipes_passed = []
    
    print(f"Testing agent: {model_path}")
    print(f"Using {'LIDAR' if use_lidar else 'Standard'} observations")
    print("-" * 70)
    print(f"{'Episode':<10} {'Score':<10} {'Pipes':<10} {'Steps':<10} {'Alive Bonus':<15}")
    print("-" * 70)
    
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0
        steps = 0
        pipe_count = 0
        last_score = 0
        
        while True:
            # Choose action
            action = agent.act(state, training=False)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            done = terminated or truncated
            
            # Count pipes (when reward jumps by ~1.0)
            if reward > 0.5:  # Pipe passed gives +1.0 reward
                pipe_count += 1
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Add delay for better visualization
            if render and delay > 0:
                time.sleep(delay)
            
            if done:
                break
        
        # Calculate alive bonus
        alive_bonus = steps * 0.1
        
        scores.append(total_reward)
        pipes_passed.append(pipe_count)
        
        print(f"{episode:<10} {total_reward:<10.2f} {pipe_count:<10} {steps:<10} {alive_bonus:<15.2f}")
    
    env.close()
    
    print("\n" + "=" * 70)
    print("Test Results Summary:")
    print("=" * 70)
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Average Pipes: {np.mean(pipes_passed):.2f} ± {np.std(pipes_passed):.2f}")
    print(f"Best Run: {np.max(pipes_passed)} pipes (Score: {scores[np.argmax(pipes_passed)]:.2f})")
    print(f"Most Consistent: Score range [{np.min(scores):.2f}, {np.max(scores):.2f}]")
    
    # Create pipe distribution
    print("\nPipe Distribution:")
    unique_pipes = sorted(set(pipes_passed))
    for p in unique_pipes:
        count = pipes_passed.count(p)
        percentage = (count / len(pipes_passed)) * 100
        print(f"  {p} pipes: {count} times ({percentage:.1f}%)")
    
    return scores, pipes_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test DDQN agent with pipe counting')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--use-lidar', action='store_true', help='Use LIDAR observations')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--delay', type=float, default=0.03, help='Delay between frames (seconds)')
    
    args = parser.parse_args()
    
    test_agent_with_pipe_count(
        args.model,
        episodes=args.episodes,
        use_lidar=args.use_lidar,
        render=not args.no_render,
        delay=args.delay
    )
