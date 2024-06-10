import random
from collections import deque

import pandas as pd
from selenium import webdriver
from selenium.common import WebDriverException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import io
import numpy as np
from PIL import Image

from agent import Agent
from state_preprocessing import preprocess_state, extract_features


class Environment:

    def __init__(self, action_time):
        self.session = None
        self.dino = None
        self._create_browser_options()
        self._action_buffer = []
        self._action_time = action_time

    def _create_browser_options(self):
        chrome_options = Options()
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--mute-audio")
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self._chrome_options = chrome_options

    def create_session(self):
        self.session = webdriver.Chrome(options=self._chrome_options)
        self.dino = self.session.find_element(By.TAG_NAME, 'body')
        try:
            self.session.get('chrome://dino')
        except WebDriverException:
            pass

    def get_state(self, flatten=True):
        screenshot = self.session.get_screenshot_as_png()
        image_values = np.array(Image.open(io.BytesIO(screenshot)))
        return preprocess_state(image_values, flatten=flatten)

    def apply_action(self, action):
        actions = ActionChains(self.session)
        if action == 'jump':
            actions.key_down(Keys.SPACE).perform()
            time.sleep(self._action_time)
            actions.key_up(Keys.SPACE).perform()
        elif action == 'duck':
            actions.key_down(Keys.DOWN).perform()
            time.sleep(self._action_time)
            actions.key_up(Keys.DOWN).perform()
        else:
            time.sleep(self._action_time)


def is_game_over(last_frame, frame):
    last_frame = last_frame.flatten()
    frame = frame.flatten()
    if last_frame is None:
        return False
    for i in range(frame.shape[0]):
        if frame[i] != last_frame[i]:
            return False
    return True


fps = 60
time_per_each_frame = 1 / fps
env = Environment(time_per_each_frame)
env.create_session()
last_frame = None
agent = Agent(4, 3, epsilon=0.8)
min_eps = 0.15
n_episodes = 1000
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.display.float_format = '{:,.5f}'.format

for episode in range(n_episodes):

    env.apply_action('jump')
    done = False
    reward = 0.2
    state_counter = 0
    reward_sum = 0
    speed_of_game = 0.01

    while not done:
        frame = env.get_state(flatten=False)
        features = extract_features(frame, speed_of_game)
        action = agent.epsilon_greedy_action(features)
        env.apply_action(['jump', 'duck', 'none'][action])
        next_frame = env.get_state(flatten=False)
        next_features = extract_features(next_frame, speed_of_game)
        if is_game_over(frame, next_frame):
            reward = -10
            done = True
            time.sleep(1)

        agent.update_q_function(features, action, reward, next_features)
        reward_sum += reward
        state_counter += 1
        speed_of_game += 0.01

    if agent.epsilon > min_eps:
        agent.epsilon -= 0.0001
    print(
        f"Episode: {episode}, eps={agent.epsilon} , counts={state_counter}, "
        f"avg-theta={np.mean(agent.theta)}, avg-reward={reward_sum / state_counter}")
