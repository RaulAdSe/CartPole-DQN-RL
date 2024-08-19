# CartPole-DQN-RL

This project implements a DQN agent from scratch to solve the CartPole balancing problem.

## Project Structure

- **environment/**: Contains the custom CartPole environment.
- **agent/**: Contains the DQN agent and replay buffer.
- **utils/**: Utility functions including the neural network and training loop.
- **main.py**: The entry point for running the training.

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/cartpole_rl.git
    cd cartpole_rl
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Training

To train the DQN agent, run:
```bash
python main.py
