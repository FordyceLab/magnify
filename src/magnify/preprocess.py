from __future__ import annotations

import os
import pathlib

import tifffile
import xarray as xr

import magnify.registry as registry


@registry.component("rotate")
def rotate(assay: xr.Dataset, rotation=0):
    # TODO: Fix issue with rotation bug in dask.
    # assay["image"].data = dask_image.ndinterp.rotate(
    #     assay.image.data, rotation, axes=(-1, -2), reshape=False
    # )
    return assay


@registry.component("flatfield_correct")
def flatfield_correct(assay: xr.Dataset, flatfield=1.0, darkfield=0.0):
    if isinstance(flatfield, os.PathLike | str):
        flatfield = pathlib.Path(flatfield).expanduser()
        if flatfield.is_dir():
            # We have a zarr directory with both flatfield and darkfield information.
            flatfield = xr.dataset.open_zarr(flatfield, group="flatfield")["flatfield"]
            # Account for channel mismatches.
            flatfield = xr.align(assay.tile, flatfield, join="left")[1].fillna(
                flatfield.sel(channel="default")
            )
            darkfield = xr.dataset.open_zarr(flatfield, group="darkfield")["darkfield"]
        else:
            with tifffile.TiffFile(flatfield) as tif:
                flatfield = tif.asarray()

    if isinstance(darkfield, os.PathLike | str):
        darkfield = pathlib.Path(darkfield).expanduser()
        with tifffile.TiffFile(darkfield) as tif:
            darkfield = tif.asarray()

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
