import numpy as np
import random
from collections import deque
from utils.neural_network import NeuralNetwork

#In the DQN (Deep Q-Network) implementation, the Q-values are not explicitly stored in a table as they would be in traditional Q-learning. Instead, they are computed dynamically by the neural network. In summary, the Q-values are "stored" implicitly in the weights of the neural network, and the process of training adjusts these weights to make the Q-values more accurate over time.
#1.Neural Network as a Function Approximator. Given a state, the neural network predicts a set of Q-values, one for each possible action.
#2.Q-Value Prediction: When the DQN agent needs to choose an action or update its knowledge, it feeds the current state (or next state) into the neural network. The neural network then outputs a vector of Q-values, where each element corresponds to a Q-value for a specific action.
#3.Updating Q-Values: During training, the agent uses these predicted Q-values to learn. It calculates a target Q-value for the action taken based on the reward received and the maximum Q-value predicted for the next state. The Q-value for the chosen action in the current state is then adjusted towards this target.
#4.Adjusting the Weights: The difference between the predicted Q-value and the target Q-value (this difference is known as the TD error) is what the model tries to minimize.  target_f is essentially a copy of the predicted Q-values, but with the Q-value for the chosen action replaced by the target value. The neural network is then trained (using gradient descent) to make its Q-value predictions closer to the target values.

# Summary of DQN Workflow
#1.Initialize the environment and the DQN agent.
#2.Interact with the environment: For each state, choose an action using the epsilon-greedy policy.
#3.Observe the reward and next state, and store this experience in memory.
#4.Replay experiences: Sample a minibatch from memory and update the Q-values using the neural network.
#5.Reduce exploration over time by decaying epsilon.
#6.Repeat until the agent has learned to maximize the reward or until the training session ends.

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = NeuralNetwork(state_size, action_size, self.learning_rate)  # Use NeuralNetwork class

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load(name)

    def save(self, name):
        self.model.save(name)