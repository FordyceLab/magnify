import numpy as np
import xarray as xr

import magnify.registry as registry
import magnify.utils as utils


@registry.component("drop")
def drop(
    assay: xr.Dataset,
    squeeze: bool = True,
    roi_only: bool = True,
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


@registry.component("summarize_sum")
def summarize_sum(assay: xr.Dataset):
    fg_area = assay.fg.sum(dim=["roi_x", "roi_y"]).compute()
    fg = assay.roi.where(assay.fg).sum(dim=["roi_x", "roi_y"]).compute()
    bg = (assay.roi.where(assay.bg).median(dim=["roi_x", "roi_y"]) * fg_area).compute()
    assay["mark_intensity"] = fg - bg

    shape = (len(assay.tag),) + assay.image.shape[:2]
    tag_intensity = xr.DataArray(data=np.empty(shape), dims=("tag", "channel", "time"))
    for i, tag in enumerate(assay.tag):
        subintensity = utils.sel_tag(fg, tag) - utils.sel_tag(bg, tag)
        if tag != "":
            subintensity = subintensity.where(utils.sel_tag(assay.valid, tag))
        tag_intensity[i] = subintensity.mean(dim="mark")

    assay["tag_intensity"] = tag_intensity
    return assay
