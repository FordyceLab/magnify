import holoviews as hv
import holoviews.operation.datashader as ds
import xarray as xr

from magnify.magniplot.ndplot import ndplot


def roishow(assay: xr.Dataset, grid=None, slider=None, rasterize=True, **kwargs):
    if grid is None and slider is None:
        slider = ["channel", "time"]
        grid = ["mark", "mark_row", "mark_col"]

    def imfunc(assay: xr.Dataset, **kwargs):
        img = hv.Image((assay.roi_x, assay.roi_y, assay.roi)).opts(**kwargs)
        return img

    img = ndplot(assay, imfunc, grid=grid, slider=slider, **kwargs)
    return ds.rasterize(img) if rasterize else img


def imshow(assay: xr.Dataset, grid=None, slider=None, rasterize=True, **kwargs):
    if grid is None and slider is None:
        slider = ["channel", "time"]

    def imfunc(assay: xr.Dataset, **kwargs):
        img = hv.Image((assay.im_x, assay.im_y, assay.image)).opts(**kwargs)
        return img

    img = ndplot(assay, imfunc, grid=grid, slider=slider, **kwargs)
    return ds.rasterize(img) if rasterize else img
