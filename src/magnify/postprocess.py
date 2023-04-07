import xarray as xr

import magnify.registry as registry


@registry.component("squeeze")
def squeeze(assay: xr.Dataset):
    return assay.squeeze(drop=True)
