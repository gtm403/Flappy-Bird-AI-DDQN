# Flappy Bird DDQN Agent

Double Deep Q-Network (DDQN) agents that learn to play Flappy Bird using the flappy_bird_gymnasium environment.

![Demo](./Screen%20Recording%202025-09-07%20091911.gif)

## Performance of Best Model

- **Best Run**: 425 pipes
- **Average Performance**: 122 pipes over 20 episodes
- **Training Time**: 5000 episodes


<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/1f9cde10-1559-4f5a-a235-4005cd1a9c10" />


## Features

- **Double DQN Algorithm**: Reduces overestimation bias compared to standard DQN
- **Prioritized Experience Replay**: Large buffer (50k-100k samples) for stable learning
- **Dual Observation Modes**: 
  - Standard (12 features) - Recommended
  - LIDAR (180 features) - Experimental
- **Detailed Performance Analysis**: Track both score and pipe count
- **Model Checkpointing**: Save best models and periodic checkpoints
- **GPU Support**: Automatic CUDA detection for faster training; if you have a CUDA GPU, install a CUDA-enabled PyTorch build

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- Gymnasium
- flappy-bird-gymnasium
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gtm403/Flappy-Bird-AI-DDQN.git
cd Flappy-Bird-AI-DDQN

#install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Hyperparameters
- Learning rate: 0.0001-0.00025
- Gamma (discount factor): 0.99
- Epsilon: 1.0 → 0.01 (decaying)
- Replay buffer size: 10,000-50,000
- Batch size: 32-64
- Target network update frequency: 500-1000 steps





