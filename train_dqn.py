import numpy as np
import matplotlib.pyplot as plt
import pickle  # For saving episode data
from environment.cartpole import CartPoleEnv
from agent.dqn_agent import DQNAgent

if __name__ == "__main__":
    env = CartPoleEnv()
    state_size = len(env.observation_space)  # Use the length of observation_space for state_size
    action_size = len(env.action_space)  # Use the length of action_space for action_size
    agent = DQNAgent(state_size, action_size)
    episodes = 500
    scores = []

    # Initialize storage for key episodes
    episode_data = {
        "first": {"states": [], "actions": []},
        "middle": {"states": [], "actions": []},
        "last": {"states": [], "actions": []}
    }

    middle_episode = episodes // 2

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        episode_states = []  # To store states in the current episode
        episode_actions = []  # To store actions in the current episode

        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # Record states and actions for key episodes
            episode_states.append(state[0])
            episode_actions.append(action)

            if done:
                scores.append(time)
                if (e + 1) % 100 == 0:  # Print every 100 episodes
                    print(f"Episode: {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
                break

            if len(agent.memory) > 64:
                agent.replay(64)

        # Save the states and actions of the first, middle, and last episodes
        if e == 0:
            episode_data["first"]["states"] = episode_states
            episode_data["first"]["actions"] = episode_actions
        elif e == middle_episode:
            episode_data["middle"]["states"] = episode_states
            episode_data["middle"]["actions"] = episode_actions
        elif e == episodes - 1:
            episode_data["last"]["states"] = episode_states
            episode_data["last"]["actions"] = episode_actions

    agent.save("dqn_cartpole.h5")

    # Save the episode data to a file
    with open("episode_data.pkl", "wb") as f:
        pickle.dump(episode_data, f)

    # Plot the scores over episodes
    plt.figure(figsize=(12, 5))
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress Over Time')
    plt.show()
