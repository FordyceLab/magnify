from holoviews import opts
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
        # Initialize image metadata.
        len_x = assay.sizes["roi_x"]
        len_y = assay.sizes["roi_y"]
        tag_points = list(
            zip(
                assay.mark_tag.to_numpy().flatten(),
                assay.x.to_numpy().flatten(),
                assay.y.to_numpy().flatten(),
            )
        )
        valid = assay.valid.to_numpy().flatten()
        bounds = [
            hv.Bounds((x - len_x / 2, y + len_y / 2, x + len_x / 2, y - len_y / 2))
            for _, x, y in tag_points
        ]

        # Overlay image, bounding boxes, and labels.
        img = hv.Image((assay.im_x, assay.im_y, assay.image))
        img *= hv.Path([b for b, v in zip(bounds, valid) if v]).opts(color="green")
        img *= hv.Path([b for b, v in zip(bounds, valid) if not v]).opts(color="red")
        img *= hv.Labels([(x, y - 0.55 * len_y, tag) for tag, x, y in tag_points])

        # Style the plot.
        img = img.opts(
            opts.Path(line_width=1),
            opts.Labels(text_font_size="8pt", text_color="white"),
        )
        return img

    img = ndplot(assay, imfunc, grid=grid, slider=slider, **kwargs)
    return ds.rasterize(img) if rasterize else img
