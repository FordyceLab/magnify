import dask.array as da
import xarray as xr
import zarr.storage


# @xr.register_dataset_accessor("mg")
@xr.register_dataarray_accessor("mg")
class MagnifyAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._tempdir = None

    def cache(self):
        if self._tempdir is None:
            self._tempdir = zarr.storage.TempStore()

        if isinstance(self._obj, xr.Dataset):
            arrays = self._obj.variables
        else:
            arrays = {"data": self._obj}

        for name, arr in arrays.items():
            if isinstance(arr.data, da.Array):
                arr.data = da.to_zarr(
                    arr.data, url=self._tempdir, component=name, return_stored=True, overwrite=True
                )

        return self._obj
