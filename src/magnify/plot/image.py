from holoviews import opts
import holoviews as hv
import holoviews.operation.datashader as ds
import xarray as xr

from magnify.plot.ndplot import ndplot


def roishow(assay: xr.Dataset, grid=None, slider=None, rasterize=True, **kwargs):
    if grid is None and slider is None:
        slider = ["channel", "time"]
        grid = ["mark"]

    def imfunc(assay: xr.Dataset, **kwargs):
        img = hv.Image((assay.roi_x, assay.roi_y, assay.roi)).opts(**kwargs)
        return img

    plot = ndplot(assay, imfunc, grid=grid, slider=slider, **kwargs)
    plot = plot.opts(opts.Image(tools=["hover"]))
    return ds.rasterize(plot) if rasterize else plot


def imshow(assay: xr.Dataset, grid=None, slider=None, rasterize=True, compression_ratio=1, **kwargs):
    if grid is None and slider is None:
        slider = ["channel", "time"]

    def imfunc(assay: xr.Dataset, **kwargs):
        # Initialize image metadata.
        len_x = assay.sizes["roi_x"]
        len_y = assay.sizes["roi_y"]
        bounds = []
        labels = []
        for _, m in assay.groupby("mark"):
            # Get the bounds of the bounding box.
            x = m.x.item()
            y = m.y.item()
            bounds.append(
                hv.Bounds((m.x - len_x / 2, m.y + len_y / 2, m.x + len_x / 2, m.y - len_y / 2))
            )

            # Get the label for the bounding box.
            tag = m.tag.item()
            row = m.mark_row.item()
            col = m.mark_col.item()
            labels.append((x, y - 0.55 * len_y, f"{tag} ({row}, {col})"))

        valid = assay.valid.to_numpy().flatten()
        img = assay.image[..., ::compression_ratio, ::compression_ratio]
        # Overlay image, bounding boxes, and labels.
        plot = hv.Image((img.im_x, img.im_y, img))
        plot *= hv.Path([b for b, v in zip(bounds, valid) if v]).opts(color="green")
        plot *= hv.Path([b for b, v in zip(bounds, valid) if not v]).opts(color="red")
        plot *= hv.Labels(labels)

        # Style the plot.
        plot = plot.opts(
            opts.Image(tools=["hover"]),
            opts.Labels(text_font_size="8pt", text_color="white"),
            opts.Path(line_width=1),
        )
        return plot

    img = ndplot(assay, imfunc, grid=grid, slider=slider, **kwargs)
    return ds.rasterize(img, line_width=1) if rasterize else img
