import random
import numpy as np


class Agent:
    def __init__(self, n_features, n_actions, learning_rate=0.01, discount=1, epsilon=0.8):
        self.n_features = n_features
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.theta = np.zeros((self.n_features, self.n_actions))

    def q_function(self, features, action):
        return np.dot(features, self.theta[:, action])

    def update_q_function(self, features, action, reward, next_features):
        q_current = self.q_function(features, action)
        q_next_max = np.max([self.q_function(next_features, a) for a in range(self.n_actions)])
        target = reward + self.discount * q_next_max
        self.theta[:, action] += self.learning_rate * (target - q_current) * features

    def epsilon_greedy_action(self, features):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            q_values = [self.q_function(features, a) for a in range(self.n_actions)]
            return np.argmax(q_values)

    def decay_eps(self):
        self.epsilon *= self.discount
