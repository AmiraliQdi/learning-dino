from selenium import webdriver
from selenium.common import WebDriverException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import io
import numpy as np
from PIL import Image

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--disable-infobars")
chrome_options.add_argument("--mute-audio")
chrome_options.add_experimental_option("useAutomationExtension", False)
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])

# Create a new Chrome session
driver = webdriver.Chrome(options=chrome_options)

# Open the Dino game URL
try:
    driver.get('chrome://dino')
except WebDriverException:
    pass
# Wait for the Dino game page to load
time.sleep(2)


# Find the body element and start the game by sending a space key
body = driver.find_element(By.TAG_NAME, 'body')
body.send_keys(Keys.SPACE)


# Function to capture screenshot
def capture_screenshot():
    screenshot = driver.get_screenshot_as_png()
    image = Image.open(io.BytesIO(screenshot))
    return np.array(image)


# Example of running the game for a few seconds and capturing screenshots
try:
    while True:
        screenshot = capture_screenshot()
        # Here you can add code to process the screenshot and extract the game state

        # Send keys to control the game
        body.send_keys(Keys.SPACE)  # Make the dino jump
        time.sleep(0.1)
except KeyboardInterrupt:
    driver.quit()
