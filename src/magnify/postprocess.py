import xarray as xr

import magnify.registry as registry


@registry.component("squeeze")
def squeeze(assay: xr.Dataset):
    return assay.squeeze(drop=True)


@registry.component("summarize_sum")
def summarize_sum(assay: xr.Dataset):
    fg_area = assay.fg.sum(dim=["roi_x", "roi_y"])
    fg = assay.roi.where(assay.fg).sum(dim=["roi_x", "roi_y"])
    bg = assay.roi.where(assay.bg).median(dim=["roi_x", "roi_y"]) * fg_area
    assay["intensity"] = (fg - bg).compute()
    return assay
