import numpy as np

#__init__: Initializes the environment with physical constants and parameters.
#reset: Resets the environment to an initial state at the start of each episode.
#step: Takes an action, updates the state, computes the reward, and checks if the episode is done. Motion equations can be found in https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html
#render: Placeholder for future rendering capabilities.

class CartPoleEnv:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        self.state = None
        self.steps_beyond_done = None

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def step(self, action):
        assert action in [0, 1], "Action must be 0 or 1"

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        self.state = (x, x_dot, theta, theta_dot)

        done = x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        done = bool(done)

        reward = 1.0 if not done else 0.0

        return np.array(self.state), reward, done

    def render(self):
        # Rendering code can be added later
        pass
