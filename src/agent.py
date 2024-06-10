import random
import time

import numpy as np
import pandas as pd


class Agent:
    def __init__(self, n_features, n_actions, learning_rate=0.001, discount=0.99, epsilon=0.50):
        self.n_features = n_features
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.theta = np.zeros((self.n_features, self.n_actions))
        self.first_update = False

    def q_function(self, features, action):
        return np.dot(features, self.theta[:, action])

    def update_q_function(self, features, action, reward, next_features):
        if self.first_update is False:
            self.first_update = True
            return
        q_current = self.q_function(features, action)
        q_next_max = np.max([self.q_function(next_features, a) for a in range(self.n_actions)])
        target = reward + (self.discount * q_next_max) - q_current
        update = self.learning_rate * target * features
        self.theta[:, action] += update

    def epsilon_greedy_action(self, features):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            q_values = [self.q_function(features, a) for a in range(self.n_actions)]
            return np.argmax(q_values)

    def decay_eps(self):
        self.epsilon *= self.discount
