import numpy as np


def overlap_stitch(tiles: np.ndarray, overlap: int = 102) -> np.ndarray:
    tiles = tiles[:, :, :-overlap, :-overlap]
    tiles = np.concatenate(tiles, axis=1)
    return np.concatenate(tiles, axis=-1)
