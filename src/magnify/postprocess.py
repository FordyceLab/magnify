import xarray as xr

import magnify.registry as registry

@registry.components.register("squeeze")
def make_squeeze():
    def squeeze(assay: xr.Dataset):
        return assay.squeeze(drop=True)

    return squeeze

