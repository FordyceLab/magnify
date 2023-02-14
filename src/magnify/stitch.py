from numpy.typing import ArrayLike
import numpy as np


def overlap_stitch(tiles: ArrayLike, overlap: int = 102) -> np.ndarray:
    tiles = np.asarray(tiles)
    tiles = tiles[:, :, :-overlap, :-overlap]
    tiles = np.concatenate(tiles, axis=1)
    return np.concatenate(tiles, axis=-1)
