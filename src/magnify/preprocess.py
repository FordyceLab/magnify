from __future__ import annotations
from typing import cast

import basicpy
import numpy as np
import xarray as xr

import magnify.registry as registry


class Preprocessor:
    def __call__(self, assay: xr.Dataset) -> xr.Dataset:
        tiles = assay.images
        flat_tiles = assay.images.reshape(-1, tiles.shape[-2], tiles.shape[-1])
        model = basicpy.basicpy.BaSiC(get_darkfield=True, smoothness_flatfield=1)
        result = cast(np.ndarray, model.fit_transform(flat_tiles, timelapse=False))
        assay.images = result.reshape(*tiles.shape)
        return assay

    @registry.components.register("preprocessor")
    def make():
        return Preprocessor()


@registry.components.register("horizontal_flip")
def make_horizontal_flip():
    def horizontal_flip(assay: xr.Dataset):
        assay.images = np.flip(assay.images, axis=-1)
        return assay

    return horizontal_flip
