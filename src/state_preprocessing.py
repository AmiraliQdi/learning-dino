import cv2
import numpy as np
from collections import deque

from matplotlib import pyplot as plt


def preprocess_state(state):
    gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

    resized = gray[150:380, 0:]

    normalized = resized / 255.0

    return normalized


def extract_features(state, checking_border=65):
    cropped_matrix = state[:, checking_border:]
    y_axis = state.shape[0]
    max_x = cropped_matrix.shape[1]
    distance = max_x
    for i in range(max_x):
        if cropped_matrix[y_axis * 3 // 4][i] != 1:
            distance = i
    return np.array([1/(distance/max_x)])


def stack_frames(frames, new_frame, is_new_episode):
    # Initialize deque with a max length of 4 frames
    if is_new_episode:
        frames = deque([np.zeros((84, 84), dtype=np.float32) for i in range(4)], maxlen=4)
        for _ in range(4):
            frames.append(new_frame)
    else:
        frames.append(new_frame)

    # Stack the frames along the third dimension (channels)
    stacked_state = np.stack(frames, axis=2)

    return frames, stacked_state
