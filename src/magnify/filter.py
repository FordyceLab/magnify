import numpy as np
import xarray as xr

import magnify.registry as registry


@registry.component("filter_expression")
def filter_expression(assay: xr.Dataset):
    if isinstance(assay.search_channel, str):
        if assay.search_channel in assay.channel:
            search_channels = [assay.search_channel]
        elif assay.search_channel == "all":
            search_channels = assay.channel
        else:
            raise ValueError(f"{assay.search_channel} is not a channel name.")
    else:
        # We're searching across multiple channels.
        search_channels = assay.search_channel

    valid = xr.zeros_like(assay.x, dtype=bool)
    for channel in search_channels:
        subassay = assay.isel(time=0).sel(channel=channel)
        # Compute the intensity differences between every pair of backgrounds on the first timestep.
        bg = subassay.roi.where(subassay.bg).median(dim=["roi_x", "roi_y"]).compute()
        bg_n = bg.to_numpy().flatten()
        diffs = bg_n[:, np.newaxis] - bg_n[np.newaxis, :]
        offdiag = np.ones_like(diffs, dtype=bool) & (~np.eye(len(diffs), dtype=bool))
        diffs = diffs[offdiag]

        # Include any markers where the fg - bg is above 5 sigma of the mean difference.
        upper_bound = 5 * diffs.std()
        fg = subassay.roi.where(subassay.fg).median(dim=["roi_x", "roi_y"]).compute()
        valid |= fg - bg > upper_bound

    assay["valid"] = valid
    return assay
