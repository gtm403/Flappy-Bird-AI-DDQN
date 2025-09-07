# Flappy Bird DDQN Agent

A Double Deep Q-Network (DDQN) implementation that learns to play Flappy Bird using reinforcement learning.

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
- **Comprehensive Training Suite**: Multiple training scripts for different objectives
- **Detailed Performance Analysis**: Track both score and pipe count
- **Model Checkpointing**: Save best models and periodic checkpoints
- **GPU Support**: Automatic CUDA detection for faster training

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- Gymnasium
- flappy-bird-gymnasium
- NumPy
- Matplotlib

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/gtm403/flappy-bird-ddqn.git
cd flappy-bird-ddqn
