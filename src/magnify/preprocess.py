from __future__ import annotations

import os
import pathlib

import dask.array as da
import tifffile
import xarray as xr

import magnify.registry as registry


@registry.component("rotate")
def rotate(xp: xr.Dataset, rotation=0):
    # TODO: Fix issue with rotation bug in dask.
    # xp["image"].data = dask_image.ndinterp.rotate(
    #     xp.image.data, rotation, axes=(-1, -2), reshape=False
    # )
    return xp


@registry.component("flatfield_correct")
def flatfield_correct(xp: xr.Dataset, flatfield=1.0, darkfield=0.0):
    if isinstance(flatfield, os.PathLike | str):
        flatfield = pathlib.Path(flatfield).expanduser()
        if flatfield.is_dir():
            # We have a zarr directory with both flatfield and darkfield information.
            flatfield = xr.dataset.open_zarr(flatfield, group="flatfield")["flatfield"]
            # Account for channel mismatches.
            flatfield = xr.align(xp.tile, flatfield, join="left")[1].fillna(
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

    tiles = (xp.tile.astype(float) - darkfield).clip(min=0)
    max_val = tiles.max()
    tiles = tiles / flatfield
    tiles = tiles * max_val / tiles.max()
    xp["tile"] = tiles.astype(xp.tile.dtype)
    return xp


@registry.component("basic_correct")
def basic_correct(xp: xr.Dataset):
    import basicpy

    # Iterate over one channel at a time to avoid memory issues.
    for channel in xp.channel:
        tiles = xp.tile.sel(channel=channel)
        # Only fit to the first timestep since variation in time won't help with the correction.
        train_tiles = tiles.isel(time=0).to_numpy().reshape(-1, tiles.shape[-2], tiles.shape[-1])
        model = basicpy.basicpy.BaSiC(get_darkfield=True, smoothness_flatfield=1)
        model.fit(train_tiles)

        # Apply the flatfield correction.
        def transform(block, model=model):
            init_shape = block.shape
            # Reshape to handle cases where each block isn't just a single tile.
            block = block.reshape(-1, block.shape[-2], block.shape[-1])
            block = model.transform(block)
            block.reshape(init_shape)
            return block

        tiles = da.map_blocks(transform, tiles.data, dtype=tiles.dtype)
        xp["tile"].loc[{"channel": channel}] = tiles

    xp.mg.cache("tile")
    return xp


@registry.component("horizontal_flip")
def horizontal_flip(xp: xr.Dataset):
    if "image" in xp:
        xp["image"] = xp.image.isel(im_x=slice(None, None, -1))
    else:
        xp["tile"] = xp.tile.isel(tile_x=slice(None, None, -1))
    return xp


@registry.component("vertical_flip")
def vertical_flip(xp: xr.Dataset):
    if "image" in xp:
        xp["image"] = xp.image.isel(im_y=slice(None, None, -1))
    else:
        xp["tile"] = xp.tile.isel(tile_y=slice(None, None, -1))
    return xp
