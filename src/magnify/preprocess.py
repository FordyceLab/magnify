import basicpy
import numpy as np


def basic_correct_tiles(tiles: np.ndarray) -> np.ndarray:
    flat_tiles = tiles.reshape(-1, tiles.shape[-2], tiles.shape[-1])
    model = basicpy.basicpy.BaSiC(get_darkfield=True, smoothness_flatfield=1)
    result = model.fit_transform(flat_tiles, timelapse=False)
    return result.reshape(*tiles.shape)
