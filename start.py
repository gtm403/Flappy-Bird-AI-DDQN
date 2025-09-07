import os
import sys
from train import train_ddqn
from test import test_agent
from compare import analyze_observation_space

def main():
    print("Flappy Bird DDQN Quick Start")
    print("=" * 60)
    print("\nOptions:")
    print("1. Quick training (500 episodes, standard observations)")
    print("2. Quick training (500 episodes, LIDAR observations)")
    print("3. Full training (2000 episodes, standard observations)")
    print("4. Full training (2000 episodes, LIDAR observations)")
    print("5. Test existing model")
    print("6. Analyze observation spaces")
    print("0. Exit")
    
    choice = input("\nSelect option (0-6): ").strip()
    
    if choice == '0':
        sys.exit(0)
    
    elif choice in ['1', '2', '3', '4']:
        episodes = 500 if choice in ['1', '2'] else 2000
        use_lidar = choice in ['2', '4']
        
        print(f"\nTraining DDQN agent...")
        print(f"Episodes: {episodes}")
        print(f"Observations: {'LIDAR' if use_lidar else 'Standard'}")
        
        render = input("Show training visualization? (y/n): ").lower() == 'y'
        
        agent, scores = train_ddqn(
            episodes=episodes,
            use_lidar=use_lidar,
            render_mode="human" if render else None,
            save_interval=100,
            plot_interval=50
        )
        
        # Test the trained agent
        print("\nTesting trained agent...")
        model_path = sorted([f for f in os.listdir('models') if f.startswith('ddqn_')])[-1]
        model_file = f"models/{model_path}/best_model.pt"
        
        test_agent(model_file, episodes=5, use_lidar=use_lidar, render=True)
    
    elif choice == '5':
        model_dirs = sorted([d for d in os.listdir('models') if d.startswith('ddqn_')])
        
        if not model_dirs:
            print("No trained models found. Please train a model first.")
            return
        
        print("\nAvailable models:")
        for i, model_dir in enumerate(model_dirs):
            print(f"{i+1}. {model_dir}")
        
        model_idx = int(input("Select model (number): ")) - 1
        model_dir = model_dirs[model_idx]
        
        use_lidar = False
        hyperparams_file = f"models/{model_dir}/hyperparams.txt"
        if os.path.exists(hyperparams_file):
            with open(hyperparams_file, 'r') as f:
                for line in f:
                    if "Use LIDAR: True" in line:
                        use_lidar = True
                        break
        
        model_file = f"models/{model_dir}/best_model.pt"
        episodes = int(input("Number of test episodes (default 10): ") or "10")
        
        test_agent(model_file, episodes=episodes, use_lidar=use_lidar, render=True)
    
    elif choice == '6':
        analyze_observation_space()
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    main()
