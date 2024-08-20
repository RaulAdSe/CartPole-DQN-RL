import pickle
import matplotlib.pyplot as plt

# Function to plot the state of the pole for comparison
def plot_episode(states, actions, title):
    positions = [state[0] for state in states]
    angles = [state[2] for state in states]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(positions, label='Cart Position')
    plt.xlabel('Timestep')
    plt.ylabel('Position')
    plt.title(f'{title} - Cart Position')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(angles, label='Pole Angle')
    plt.xlabel('Timestep')
    plt.ylabel('Angle (radians)')
    plt.title(f'{title} - Pole Angle')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    # Load the episode data from the file
    with open("episode_data.pkl", "rb") as f:
        episode_data = pickle.load(f)

    # Visualize the first, middle, and last episodes
    plot_episode(episode_data["first"]["states"], episode_data["first"]["actions"], 'First Episode')
    plot_episode(episode_data["middle"]["states"], episode_data["middle"]["actions"], 'Middle Episode')
    plot_episode(episode_data["last"]["states"], episode_data["last"]["actions"], 'Last Episode')
