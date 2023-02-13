import numpy as np

def to_uint8(arr: np.ndarray):
    arr = arr.astype(float)
    arr = 255 * arr / np.max(arr)
    return arr.astype(np.uint8)
