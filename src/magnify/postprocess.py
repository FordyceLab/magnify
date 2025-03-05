from __future__ import annotations

import cv2 as cv
import numpy as np
import xarray as xr

import magnify.registry as registry
import magnify.utils as utils


@registry.component("drop")
def drop(
    assay: xr.Dataset,
    squeeze: bool = True,
    roi_only: bool = False,
    drop_tiles: bool = True,
):
    # TODO: We need to restore the original order.
    assay = assay.unstack()

    if squeeze:
        assay = assay.squeeze(drop=True)

    if roi_only:
        return assay.roi
    elif drop_tiles:
        return assay.drop_vars(["tile", "tile_row", "tile_col"], errors="ignore")
    else:
        return assay
