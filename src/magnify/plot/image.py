from holoviews import opts
import cv2 as cv
import holoviews as hv
import holoviews.operation.datashader as ds
import numpy as np
import xarray as xr

from magnify.plot.ndplot import ndplot
import magnify.utils as utils


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


def imshow(
    assay: xr.Dataset,
    grid=None,
    slider=None,
    rasterize=False,
    compression_ratio=1,
    contour_type="fg",
    show_centers=True,
    **kwargs,
):
    if grid is None and slider is None:
        slider = ["channel", "time"]

    def imfunc(assay: xr.Dataset, **kwargs):
        # Initialize image metadata.
        len_x = assay.sizes["roi_x"]
        len_y = assay.sizes["roi_y"]
        contours = []
        labels = []
        for idx, m in assay.groupby("mark"):
            # Get the centers and the bounds of the bounding box.
            top, bottom, left, right = utils.bounding_box(
                m.x, m.y, len_x, assay.sizes["im_x"], assay.sizes["im_y"]
            )
            # Contours are either roi bounding boxes or contours around the foreground.
            if contour_type == "roi":
                contours.append(hv.Bounds((left, bottom, right, top)))
            elif contour_type == "fg":
                # First figure out the contours based on the foreground within the ROI.
                fg_contours, _ = cv.findContours(
                    m.fg.to_numpy().astype("uint8"),
                    cv.RETR_EXTERNAL,
                    cv.CHAIN_APPROX_SIMPLE,
                )
                for c in fg_contours:
                    # Remove the extra dimension inserted by opencv.
                    c = c[:, 0]
                    # Adjust contours to be in image coordinates.
                    c = np.array(c) + [left, top]
                    # Close the curve.
                    c = np.append(c, [c[0]], axis=0)
                    contours.append(c)

            # Get the label for the bounding box.
            if "tag" in m:
                tag = m.tag.item()
            else:
                tag = ""
            if "mark_row" in m:
                row = m.mark_row.item()
                col = m.mark_col.item()
                labels.append((m.x, m.y - 0.05 * len_y, f"{tag} ({row}, {col})"))
            else:
                labels.append((m.x, m.y - 0.05 * len_y, f"{tag} ({idx})"))

        valid = assay.valid.to_numpy().flatten()
        img = assay.image[..., ::compression_ratio, ::compression_ratio]
        # Overlay image, bounding boxes, and labels.
        plot = hv.Image((img.im_x, img.im_y, img))
        plot *= hv.Path([b for b, v in zip(contours, valid) if v]).opts(color="green")
        plot *= hv.Path([b for b, v in zip(contours, valid) if not v]).opts(color="red")
        if show_centers:
            plot *= hv.Points((assay.x, assay.y)).opts(size=2, color="red")
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
