import random
from collections import deque

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
from state_preprocessing import preprocess_state, stack_frames, extract_features


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

    def get_state(self):
        screenshot = self.session.get_screenshot_as_png()
        image_values = np.array(Image.open(io.BytesIO(screenshot)))
        return preprocess_state(image_values)

    def apply_action(self, action):
        actions = ActionChains(self.session)
        if action == 'jump':
            actions.key_down(Keys.SPACE).perform()
            time.sleep(self._action_time)  # Adjust duration as needed
            actions.key_up(Keys.SPACE).perform()
        elif action == 'duck':
            actions.key_down(Keys.DOWN).perform()
            time.sleep(self._action_time)  # Hold the "duck" action for 0.5 seconds (adjust as needed)
            actions.key_up(Keys.DOWN).perform()
        else:
            pass


def rl_agent(state):
    actions = ['none']
    action = random.choice(actions)
    return action


def is_game_over(last_frame, frame):
    if last_frame is None:
        return False
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i][j] != last_frame[i][j]:
                return False
    return True


fps = 200
time_per_each_frame = 1 / fps
env = Environment(time_per_each_frame)
env.create_session()
max_frame_stack = 4
frames_stack = []
last_frame = None
agent = Agent(1, 3)
min_eps = 0.05
n_episodes = 1000

for episode in range(n_episodes):

    env.apply_action('jump')
    done = False
    reward = 0.1
    state_counter = 0

    while not done:
        frame = env.get_state()
        features = extract_features(frame)
        #print(f'dist={features[0]}')
        action = agent.epsilon_greedy_action(features)
        env.apply_action(['jump', 'duck', 'none'][action])
        next_frame = env.get_state()
        next_features = extract_features(next_frame)
        if is_game_over(frame, next_frame):
            reward = -5
            done = True
            time.sleep(1)
        agent.update_q_function(features, action, reward, next_features)
        state_counter += 1

    if agent.epsilon > min_eps:
        agent.epsilon -= 0.05

    print(f"Episode: {episode}, eps={agent.epsilon} , counts={state_counter}, theta={agent.theta}")
