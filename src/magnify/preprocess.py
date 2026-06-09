import os
import pathlib

import dask.array as da
import numpy as np
import tifffile
import xarray as xr
from scipy.ndimage import gaussian_filter

from magnify import registry, utils


@registry.component("standardize_format")
def standardize_format(xp: xr.Dataset | xr.DataArray) -> xr.Dataset:
    if isinstance(xp, xr.DataArray):
        xp = xr.Dataset({"tile": xp}).assign_attrs(xp.attrs)

    # Rename dimensions since we'll be adding new arrays whose names will conflict.
    for old_name in ["x", "y", "row", "col"]:
        if old_name in xp.tile.dims:
            xp = xp.rename({old_name: "tile_" + old_name})

    # Save the original dimension ordering to be restored later.
    xp.attrs["__original_tile_dims__"] = list(xp.tile.dims)

    desired_order = ["channel", "time", "tile_row", "tile_col", "tile_y", "tile_x"]
    # If we have additional dimensions stack them all into a single time dimension.
    extra_dims = [dim for dim in xp.tile.dims if dim not in desired_order]
    if len(extra_dims) > 0:
        if "time" in xp.tile.dims:
            # Rename the time dimension to avoid conflicts.
            xp = xp.rename(time="__time__")
            extra_dims.append("__time__")
        xp = xp.stack(time=extra_dims)

    # Reorder the dimensions so they're always consistent and add missing dimensions.
    for dim in desired_order:
        if dim not in xp.tile.dims:
            xp["tile"] = xp.tile.expand_dims(dim)

    xp = xp.transpose(*desired_order)

    return xp


@registry.component("rename_labels")
def rename_labels(xp: xr.Dataset, **coords):
    for coord_name, new_labels in coords.items():
        if isinstance(new_labels, dict):
            xp = xp.assign_coords({coord_name: xp[coord_name].to_series().replace(new_labels)})
        else:
            xp = xp.assign_coords({coord_name: new_labels})
    return xp


@registry.component("rotate")
def rotate(xp: xr.Dataset, rotation=0):
    # xp["image"].data = dask_image.ndinterp.rotate(
    #     xp.image.data, rotation, axes=(-1, -2), reshape=False
    # )
    return xp


@registry.component("flatfield_correct")
def flatfield_correct(xp: xr.Dataset, flatfield=1.0, darkfield=0.0,
                      smooth_sigma=0.0):
    """Per-tile flatfield (and optional darkfield) correction.

    Applies the standard correction::

        corrected = (tile − darkfield) / (flatfield / mean(flatfield))

    The flatfield is normalized by its own mean so the correction preserves
    overall brightness — a flatfield equal to its mean everywhere is a no-op.
    Negative values after darkfield subtraction are clipped to 0, and final
    values are clipped to the original dtype range before casting back so
    integer wraparound can't occur.

    Parameters
    ----------
    xp :
        Dataset with a ``tile`` data variable. Expected to have ``tile_y`` and
        ``tile_x`` dims (the broadcast targets); other dims may be present.
    flatfield :
        Illumination reference. Accepts:
          - a scalar (1.0 = no-op),
          - an ``xr.DataArray`` with ``tile_y, tile_x`` dims,
          - a 2-D numpy array of shape ``(tile_y, tile_x)``,
          - a path to a TIFF file containing a 2-D image.
    darkfield :
        Camera dark reference. Same accepted types as ``flatfield``. Default
        0.0 means no darkfield subtraction.
    smooth_sigma :
        If > 0, smooth the 2-D flatfield with a Gaussian filter of this sigma
        (in pixels) *before* normalizing and applying it. Suppresses
        high-frequency acquisition noise and local artifacts (dust, dye-puddle
        features) in the flatfield reference, leaving just the broad
        illumination gradient. Typical values: 30–100 px. Has no effect when
        ``flatfield`` is a scalar. Default 0 = no smoothing.
    """
    ty, tx = xp.sizes["tile_y"], xp.sizes["tile_x"]

    def _resolve(arg, name):
        # Load TIFF → numpy array.
        if isinstance(arg, (os.PathLike, str)):
            arg = tifffile.imread(pathlib.Path(arg).expanduser())
        # Already a labeled DataArray: trust the caller's dim names.
        if isinstance(arg, xr.DataArray):
            return arg
        # 2-D numpy array: wrap with explicit dim names so xarray broadcasts
        # against the right axes regardless of `xp.tile`'s dim order.
        if hasattr(arg, "ndim") and arg.ndim == 2:
            if arg.shape != (ty, tx):
                raise ValueError(
                    f"{name} shape {arg.shape} does not match tile "
                    f"shape (tile_y={ty}, tile_x={tx})"
                )
            return xr.DataArray(arg, dims=("tile_y", "tile_x"))
        # Scalar (or unsupported type passed through): return as-is; xarray
        # broadcasts scalars cleanly. Non-scalar non-2D inputs will surface
        # via downstream arithmetic errors.
        return arg

    flatfield = _resolve(flatfield, "flatfield")
    darkfield = _resolve(darkfield, "darkfield")

    # Optionally smooth the flatfield to suppress acquisition noise / local
    # artifacts (dust, dye-puddle features) while keeping the broad
    # illumination gradient. Applied BEFORE the mean-normalization so the
    # normalization uses the smoothed reference.
    if isinstance(flatfield, xr.DataArray) and smooth_sigma and smooth_sigma > 0:
        smoothed = gaussian_filter(
            np.asarray(flatfield.values, dtype=np.float32),
            sigma=float(smooth_sigma),
        )
        flatfield = xr.DataArray(smoothed, dims=flatfield.dims,
                                 coords=flatfield.coords)

    # Normalize flatfield to mean=1 so the correction preserves overall
    # brightness (only spatial non-uniformity is corrected).
    if isinstance(flatfield, xr.DataArray):
        ff_mean = float(flatfield.mean())
        if ff_mean == 0:
            raise ValueError("flatfield mean is zero — cannot normalize")
        flatfield = flatfield / ff_mean

    tiles = (xp.tile.astype("float32") - darkfield).clip(min=0)
    tiles = tiles / flatfield

    # Clip to the original dtype range before casting back so integer
    # wraparound (e.g. uint16 overflow) can't quietly corrupt the result.
    orig_dtype = xp.tile.dtype
    if orig_dtype.kind in "iu":
        info = np.iinfo(orig_dtype)
        tiles = tiles.clip(min=info.min, max=info.max)
    xp["tile"] = tiles.astype(orig_dtype)
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
            return block.reshape(init_shape)

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


@registry.component("circle_mask")
def circle_mask(
    xp: xr.Dataset,
    center: tuple[int, int],
    diameter: int,
    mask_inner=False,
):
    radius = diameter // 2
    img_shape = (xp.image.shape if "image" in xp else xp.tile.shape)[-2:]

    mask = utils.circle(img_shape, center, radius, True)
    mask = ~mask if mask_inner else mask
    if "image" in xp:
        xp["image"] *= mask
    else:
        xp["tile"] *= mask

    return xp
