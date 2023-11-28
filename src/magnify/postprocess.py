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
    if squeeze:
        assay = assay.squeeze(drop=True)

    if roi_only:
        return assay.roi
    elif drop_tiles:
        return assay.drop_vars("tile")
    else:
        return assay


@registry.component("circularize")
def circularize(
    assay: xr.Dataset,
):
    subassay = assay.isel(time=0, channel=0)
    fg = subassay.fg.to_numpy()
    bg = subassay.bg.to_numpy()
    for i in range(assay.sizes["mark"]):
        old_fg = utils.to_uint8(fg[i])
        contour = cv.findContours(old_fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]
        center, radius = cv.minEnclosingCircle(contour)
        fg[i] = utils.circle(
            fg.shape[-1],
            row=round(center[1]),
            col=round(center[0]),
            radius=round(radius),
            value=True,
        )
        bg[i] = ~fg[i] & bg[i]
    assay.fg[:] = fg[:, np.newaxis, np.newaxis, :, :]
    assay.bg[:] = bg[:, np.newaxis, np.newaxis, :, :]
    return assay
