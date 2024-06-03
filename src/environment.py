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

from state_preprocessing import preprocess_state, stack_frames


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


def if_game_over(last_frame, frame):
    if last_frame is None:
        return False
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i][j] != last_frame[i][j]:
                return False
    return True


fps = 20
time_per_each_frame = 1 / fps
env = Environment(time_per_each_frame)
env.create_session()
max_frame_stack = 4
frames = deque(maxlen=max_frame_stack)
state_counter = 0
last_frame = None

try:
    is_new_episode = True
    while True:

        if state_counter % max_frame_stack == 0:
            is_new_episode = True
        else:
            is_new_episode = False

        frame = env.get_state()

        if if_game_over(last_frame, frame):
            print("GAME OVER")

        last_frame = frame

        frames, stacked_state = stack_frames(frames, frame, is_new_episode)

        action = rl_agent(stacked_state)
        state_counter += 1

        env.apply_action(action)


except KeyboardInterrupt:
    env.session.quit()
