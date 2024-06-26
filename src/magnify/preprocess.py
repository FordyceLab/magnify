from __future__ import annotations
from typing import cast
import os

import dask_image.ndinterp
import tifffile
import numpy as np
import xarray as xr

import magnify.registry as registry


@registry.component("rotate")
def rotate(assay: xr.Dataset, rotation=0):
    print(assay["image"])
    assay["image"].data = dask_image.ndinterp.rotate(assay.image.data, rotation, axes=(-1, -2), reshape=False)
    import matplotlib.pyplot as plt
    print(assay["image"].data.dtype)
    plt.imshow(assay["image"].to_numpy()[0, 0])
    return assay


@registry.component("flatfield_correct")
def flatfield_correct(assay: xr.Dataset, flatfield=1.0, darkfield=0.0):
    if isinstance(flatfield, os.PathLike):
        with tifffile.TiffFile(os.path.expanduser(flatfield)) as tif:
            flatfield = tif.asarray()
    if isinstance(darkfield, os.PathLike):
        with tifffile.TiffFile(os.path.expanduser(darkfield)) as tif:
            darkfield = tif.asarray()

    dtype = assay.tile.dtype
    tiles = (assay.tile.astype(float) - darkfield).clip(min=0)
    max_val = tiles.max()
    tiles = tiles / flatfield
    tiles = tiles * max_val / tiles.max()
    assay["tile"] = tiles.astype(assay.tile.dtype)
    return assay


@registry.component("horizontal_flip")
def horizontal_flip(assay: xr.Dataset):
    if "image" in assay:
        assay["image"] = assay.image.isel(im_x=slice(None, None, -1))
    else:
        assay["tile"] = assay.tile.isel(tile_x=slice(None, None, -1))
    return assay


@registry.component("vertical_flip")
def vertical_flip(assay: xr.Dataset):
    if "image" in assay:
        assay["image"] = assay.image.isel(im_y=slice(None, None, -1))
    else:
        assay["tile"] = assay.tile.isel(tile_y=slice(None, None, -1))
    return assay
