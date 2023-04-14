import holoviews as hv
import holoviews.operation.datashader as ds
import xarray as xr

from magnify.magniplot.ndplot import ndplot


def roishow(assay: xr.Dataset, grid_dims=None, slider_dims=None, rasterize=True, **kwargs):
    if slider_dims is None:
        slider_dims = ["channel", "time"]
    if grid_dims is None:
        grid_dims = ["marker", "marker_row", "marker_col"]

    def imfunc(assay: xr.Dataset, **kwargs):
        img = hv.Image((assay.roi_x, assay.roi_y, assay.roi)).opts(**kwargs)
        return img

    img = ndplot(assay, imfunc, grid_dims=grid_dims, slider_dims=slider_dims, **kwargs)
    return ds.rasterize(img) if rasterize else img


def imshow(assay: xr.Dataset, grid_dims=None, slider_dims=None, rasterize=True, **kwargs):
    if slider_dims is None:
        slider_dims = ["channel", "time"]

    def imfunc(assay: xr.Dataset, **kwargs):
        img = hv.Image((assay.im_x, assay.im_y, assay.image)).opts(**kwargs)
        return img

    img = ndplot(assay, imfunc, grid_dims=grid_dims, slider_dims=slider_dims, **kwargs)
    return ds.rasterize(img) if rasterize else img
