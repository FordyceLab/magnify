from __future__ import annotations

import xarray as xr


def save(file, xp):
    # xarrays don't support saving multi-indices so unstack before saving.
    xp.unstack().to_netcdf(file)


def load(file):
    xp = xr.open_dataset(file)
    # If we have a chip dataset we want to restack the mark_col and mark_row indices
    # which we unstacked before saving.
    if "mark_row" in xp.dims and "mark_col" in xp.dims:
        xp = xp.stack(mark=("mark_row", "mark_col")).transpose("mark", ...)
    return xp
