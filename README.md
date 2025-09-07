# Flappy Bird DDQN Agent

A high-performance Double Deep Q-Network (DDQN) implementation that learns to play Flappy Bird using reinforcement learning. The agent achieves superhuman performance with a record of **425 pipes** and averages over 100 pipes per game.

![Demo](./Screen%20Recording%202025-09-07%20091911.gif)

## Performance

- **Best Run**: 425 pipes
- **Average Performance**: 122 pipes
- **Training Time**: ~10 minutes for 5000 episodes
- **Peak Performance Episode**: 4890 (109 pipes during training)



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
git clone https://github.com/yourusername/flappy-bird-ddqn.git
cd flappy-bird-ddqn
