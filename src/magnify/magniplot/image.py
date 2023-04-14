import holoviews as hv
import xarray as xr
from holoviews.operation.datashader import rasterize

from magnify.magniplot.layout import multidim


@multidim(grid_dims=["marker", "marker_row", "marker_col"], slider_dims=["time", "channel"])
def roi_image(assay: xr.Dataset, **kwargs):
    img = hv.Image((assay.roi_x, assay.roi_y, assay.roi)).opts(**kwargs)
    return img

@multidim(slider_dims=["time", "channel"])
def image(assay: xr.Dataset, **kwargs):
    img = hv.Image((assay.im_x, assay.im_y, assay.image)).opts(**kwargs)
    return img
