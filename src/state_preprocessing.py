import cv2
import numpy as np
from collections import deque


def preprocess_state(state):
    # Convert to grayscale
    gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

    # Resize the image
    resized = cv2.resize(gray, (84, 84))

    # Normalize pixel values to [0, 1]
    normalized = resized / 255.0

    return normalized


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


