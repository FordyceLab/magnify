from __future__ import annotations
from typing import cast

import basicpy
import numpy as np
import xarray as xr

import magnify.registry as registry


class Preprocessor:
    def __call__(self, assay: xr.Dataset) -> xr.Dataset:
        tiles = assay.image
        flat_tiles = assay.image.reshape(-1, tiles.dim["im_row"], tiles.dim["im_col"])
        model = basicpy.basicpy.BaSiC(get_darkfield=True, smoothness_flatfield=1)
        result = cast(np.ndarray, model.fit_transform(flat_tiles, timelapse=False))
        assay = assay.assign(image=result.reshape(*tiles.shape))
        return assay

    @registry.components.register("preprocessor")
    def make():
        return Preprocessor()


@registry.components.register("horizontal_flip")
def make_horizontal_flip():
    def horizontal_flip(assay: xr.Dataset):
        assay["image"] = assay.image.isel(im_col=slice(None, None, -1))
        return assay

    return horizontal_flip
