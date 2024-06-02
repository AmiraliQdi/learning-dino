import threading

import pygame
from pygame import *

from src.game import start_game


class Environment:

    def __init__(self):
        self.game_thread = threading.Thread(name="game_thread", target=start_game)

    def start_game(self):
        self.game_thread.start()


env = Environment()
env.start_game()

