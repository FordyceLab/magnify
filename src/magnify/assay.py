import numpy as np


class Assay:
    type: str
    ids: np.ndarray
    times: np.ndarray
    channels: list[str]
    info: dict

    images: np.ndarray
    regions: np.ndarray
    offsets: np.ndarray
    centers: np.ndarray
    fg: np.ndarray
    bg: np.ndarray
    valid: np.ndarray
