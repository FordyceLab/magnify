from __future__ import annotations
from typing import cast
import os

import basicpy
import tifffile
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

    @registry.components.register("preprocess")
    def make():
        return Preprocessor()


@registry.components.register("flatfield_correct")
def make_flatfield_correct():
    def flatfield_correct(assay: xr.Dataset, flatfield=1.0, darkfield=0.0):
        if isinstance(flatfield, str):
            with tifffile.TiffFile(os.path.expanduser(flatfield)) as tif:
                flatfield = tif.asarray()
        if isinstance(darkfield, str):
            with tifffile.TiffFile(os.path.expanduser(darkfield)) as tif:
                darkfield = tif.asarray()

        assay["image"] = (assay.image - darkfield) / flatfield
        return assay

    return flatfield_correct


@registry.components.register("flip_horizontal")
def make_horizontal_flip():
    def horizontal_flip(assay: xr.Dataset):
        assay["image"] = assay.image.isel(im_col=slice(None, None, -1))
        return assay

    return horizontal_flip
