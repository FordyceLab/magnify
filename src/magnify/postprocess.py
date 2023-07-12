from __future__ import annotations

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
    if squeeze:
        assay = assay.squeeze(drop=True)

    if roi_only:
        return assay.roi
    elif drop_tiles:
        return assay.drop_vars("tile")
    else:
        return assay
