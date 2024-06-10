import time

import cv2
import numpy as np
from collections import deque

import pandas as pd
from matplotlib import pyplot as plt


# y-len=187
def preprocess_state(state, flatten=True):
    resized = state[173:350, 50:]
    grayscale_matrix = np.dot(resized[..., :3], [0.2989, 0.5870, 0.1140])

    # Apply a threshold to convert grayscale to binary (black or white)
    # Here we consider a pixel black if its grayscale value is below a certain threshold
    threshold = 128
    binary_matrix = (grayscale_matrix >= threshold).astype(int)

    # Flatten the 2D binary matrix to a 1D binary array
    if flatten:
        return binary_matrix.flatten()
    else:
        return binary_matrix


def extract_features(state, speed_of_game):
    feature_list = []
    x_ob, y_ob = find_nearest_obstacle(state)
    feature_list.append(x_ob)
    feature_list.append(y_ob)
    dino_height = find_dino_height(state)
    feature_list.append(dino_height)
    feature_list.append(speed_of_game)
    return np.array(feature_list)


def find_dino_height(state):
    cropped = state[:, :50]
    # plt.imshow(cropped[:, :], cmap='gray')
    # plt.title('Binary Image')
    # plt.axis('off')  # Hide axis
    # plt.show()
    for y in range(cropped.shape[0]//2):
        if cropped[y][cropped.shape[1] // 2] == 1:
            return 0
    return 1


def find_nearest_obstacle(state, x_checking_border=55):
    state = state[:160, 65:]
    max_x = state.shape[1]
    max_y = state.shape[0]
    for x in range(x_checking_border, max_x):
        for y in range(max_y):
            if state[y][x] == 1:
                return 1 / x, 1 / (max_y - y)
    return 1 / max_x, 0
