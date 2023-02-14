import numpy as np

class Assay:
    type: str
    ids: np.ndarray
    subimage_offsets: np.ndarray
    centers: np.ndarray

    images: np.ndarray
    subimages: np.ndarray
    fg_values: np.ndarray
    bg_values: np.ndarray
    valid: np.ndarray
    channels: np.ndarray
