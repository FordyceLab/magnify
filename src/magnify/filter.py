from __future__ import annotations

import cv2 as cv
import numpy as np
import xarray as xr

import magnify.registry as registry
from magnify import utils


@registry.component("filter_expression")
def filter_expression(
    assay: xr.Dataset,
    search_channel: str | list[str] | None = None,
    min_contrast: int | None = None,
):
    if search_channel is None:
        search_channels = assay.channel
    else:
        search_channels = utils.to_list(search_channel)

    valid = xr.zeros_like(assay.valid, dtype=bool)
    for channel in search_channels:
        subassay = assay.isel(time=0).sel(channel=channel)
        fg = subassay.roi.where(subassay.fg).median(dim=["roi_x", "roi_y"]).compute()
        bg = subassay.roi.where(subassay.bg).median(dim=["roi_x", "roi_y"]).compute()
        if min_contrast is None:
            # Compute the intensity differences between every pair of backgrounds on the first timestep.
            bg_n = bg.to_numpy().flatten()
            diffs = bg_n[:, np.newaxis] - bg_n[np.newaxis, :]
            offdiag = np.ones_like(diffs, dtype=bool) & (~np.eye(len(diffs), dtype=bool))
            diffs = diffs[offdiag]

            # Include any markers where the fg - bg is above 5 sigma of the mean difference.
            upper_bound = 4 * diffs.std()
        else:
            upper_bound = min_contrast
        valid |= fg - bg > upper_bound

    assay["valid"] &= valid
    return assay


@registry.component("filter_nonround")
def filter_nonround(
    assay: xr.Dataset, min_roundness: float = 0.75, search_channel: str | list[str] | None = None
):
    if search_channel is None:
        search_channels = assay.channel
    else:
        search_channels = utils.to_list(search_channel)

    valid = assay.valid.to_numpy()
    for channel in search_channels:
        subassay = assay.isel(time=0).sel(channel=channel)
        fg = utils.to_uint8(subassay.fg.to_numpy())
        areas = subassay.fg.sum(dim=["roi_x", "roi_y"]).to_numpy()
        for i in range(assay.sizes["mark"]):
            contours, _ = cv.findContours(fg[i], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            perimeter = sum(cv.arcLength(c, True) for c in contours)
            if perimeter == 0:
                assay.valid[i] = False
                continue
            roundness = 4 * np.pi * float(areas[i]) / perimeter**2
            valid[i] &= roundness > min_roundness
    assay.valid.data = valid

    return assay


@registry.component("filter_leaky")
def filter_leaky_buttons(assay: xr.Dataset, search_channel: str | list[str] | None = None):
    if search_channel is None:
        search_channels = assay.channel
    else:
        search_channels = utils.to_list(search_channel)

    tag = assay.tag.to_numpy()
    valid = assay.valid.to_numpy()
    rows = assay.mark_row.to_numpy()
    for channel in search_channels:
        sub_roi = assay.sel(channel=channel, time=0).roi.compute()
        # Compute the intensity differences between every pair of backgrounds on the first timestep.
        bg = sub_roi.where(sub_roi.bg).median(dim=["roi_x", "roi_y"]).compute()
        bg_n = bg.to_numpy().flatten()
        diffs = bg_n[:, np.newaxis] - bg_n[np.newaxis, :]
        offdiag = np.ones_like(diffs, dtype=bool) & (~np.eye(len(diffs), dtype=bool))
        diffs = diffs[offdiag]

        # Find markers where the fg - bg is below 5 sigma of the mean difference.
        upper_bound = 5 * diffs.std()
        fg = sub_roi.where(sub_roi.fg).median(dim=["roi_x", "roi_y"])
        empty = (fg - bg < upper_bound).to_numpy()
        for i in range(assay.sizes["mark"]):
            row = rows[i]
            if tag[i] == "":
                continue
            if row > 0 and tag[i - 1] == "":
                valid[i] &= empty[i - 1]
            if row < assay.mark_row.max() and tag[i + 1] == "":
                valid[i] &= empty[i + 1]
    assay.valid.data = valid

    return assay
