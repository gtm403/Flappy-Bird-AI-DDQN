import gymnasium
import flappy_bird_gymnasium
import numpy as np
from agent import DDQNAgent
import time
import argparse

def test_agent(model_path, episodes=10, use_lidar=False, render=True, delay=0.03):
    
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
    
    print(f"Testing agent: {model_path}")
    print(f"Using {'LIDAR' if use_lidar else 'Standard'} observations")
    print("-" * 50)
    
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0
        steps = 0
        
        while True:
            # Choose action
            action = agent.act(state, training=False)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if render and delay > 0:
                time.sleep(delay)
            
            if done:
                break
        
        scores.append(total_reward)
        print(f"Episode {episode}: Score = {total_reward:.2f}, Steps = {steps}")
    
    env.close()
    
    print("\nTest Results:")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Best Score: {np.max(scores):.2f}")
    print(f"Worst Score: {np.min(scores):.2f}")
    
    return scores

def play_interactive():
    print("Interactive Mode - Press SPACE to flap, Q to quit")
    print("This mode is for human play comparison")
    
    env = gymnasium.make("FlappyBird-v0", render_mode="human")
    
    # Note: Gymnasium Flappy Bird uses action space, not keyboard events
    # This is just to show how a human would score
    print("\nNote: The gymnasium environment doesn't support keyboard input directly.")
    print("Showing random agent for comparison. Run with --test to see trained agent.\n")
    
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    
    while True:
        # Random action for demonstration
        action = env.action_space.sample()
        
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        steps += 1
        
        time.sleep(0.03)
        
        if done:
            break
    
    env.close()
    print(f"\nGame Over! Score: {total_reward:.2f}, Steps: {steps}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test DDQN agent on Flappy Bird')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--use-lidar', action='store_true', help='Use LIDAR observations')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--delay', type=float, default=0.03, help='Delay between frames (seconds)')
    parser.add_argument('--interactive', action='store_true', help='Play interactively (demo mode)')
    
    args = parser.parse_args()
    
    if args.interactive:
        play_interactive()
    elif args.model:
        test_agent(
            args.model,
            episodes=args.episodes,
            use_lidar=args.use_lidar,
            render=not args.no_render,
            delay=args.delay
        )
    else:
        print("Please provide a model path with --model or use --interactive for demo mode")
        parser.print_help()
