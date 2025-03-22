import tempfile

import dask.array as da
import xarray as xr

from magnify import utils

cache = []


@xr.register_dataset_accessor("mg")
class MagnifyAccessor:
    def __init__(self, dataset):
        self._dataset = dataset
        self._tempdir = tempfile.TemporaryDirectory()
        cache.append(self._tempdir)

    def cache(self, variables=None):
        variables = utils.to_list(variables)
        if variables:
            arrays = {name: self._dataset.variables[name] for name in variables}
        else:
            arrays = self._dataset.variables

        for name, arr in arrays.items():
            if isinstance(arr.data, da.Array):
                arr.data = da.to_zarr(
                    arr.data,
                    url=self._tempdir.name,
                    component=name,
                    return_stored=True,
                    overwrite=True,
                )

        return self._dataset
