from __future__ import annotations

import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr

from magnify.plot.ndplot import ndplot
import magnify.utils as utils


def roishow(
    xp: xr.Dataset,
    facet_col=None,
    animation_frame=None,
    binary_string=True,
    binary_format="jpeg",
    binary_compression_level=0,
    cmap="viridis",
    zmin=None,
    zmax=None,
    **kwargs,
):
    def imfunc(xp: xr.Dataset, **kwargs):
        img = xp.roi.compute()
        if binary_string:
            img = mpl.colors.Normalize(vmin=zmin, vmax=zmax)(img.to_numpy())
            img = plt.get_cmap(cmap)(img)[:, :, :3]
        fig = px.imshow(img, binary_string=binary_string, binary_format=binary_format)
        contours = get_contours(xp)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([c[:, 0] for c in contours]),
                y=np.concatenate([c[:, 1] for c in contours]),
                mode="lines",
                showlegend=False,
                line_color="green",
            )
        )
        return fig.data

    fig = ndplot(xp, imfunc, animation_frame=animation_frame, facet_col=facet_col, **kwargs)
    fig.update_layout(width=800, height=800, dragmode="pan")
    return fig


def imshow(
    xp: xr.Dataset,
    facet_col=None,
    animation_frame=None,
    binary_string=True,
    binary_format="jpeg",
    binary_compression_level=0,
    compression_ratio=1,
    contour_type="roi",
    cmap="viridis",
    zmin=None,
    zmax=None,
    **kwargs,
):
    def imfunc(xp: xr.Dataset, **kwargs):
        img = xp.image[..., ::compression_ratio, ::compression_ratio].compute()
        if binary_string:
            img = mpl.colors.Normalize(vmin=zmin, vmax=zmax)(img.to_numpy())
            img = plt.get_cmap(cmap)(img)[:, :, :3]
        fig = px.imshow(img, binary_string=binary_string, binary_format=binary_format)
        if "roi" in xp:
            roi = xp.roi.compute()
            # Initialize image metadata.
            roi_len = roi.sizes["roi_y"] // compression_ratio
            valid_x = []
            valid_y = []
            valid_labels = []
            invalid_x = []
            invalid_y = []
            invalid_labels = []
            for idx, m in roi.groupby("mark"):
                x = m.x.item() / compression_ratio
                y = m.y.item() / compression_ratio
                # Get the centers and the bounds of the bounding box.
                top, bottom, left, right = utils.bounding_box(
                    x, y, roi_len, img.shape[1], img.shape[0]
                )
                # Contours are either roi bounding boxes or contours around the foreground.
                if contour_type == "roi":
                    contour_x = [left, left, right, right, left, None]
                    contour_y = [bottom, top, top, bottom, bottom, None]
                elif contour_type == "fg":
                    cont = get_contours(m)
                    # Adjust contours to be in image coordinates.
                    contour_x = list(np.concatenate([c[:, 0] + left for c in cont])) + [None]
                    contour_y = list(np.concatenate([c[:, 1] + top for c in cont])) + [None]

                # Get the label for the bounding box.
                if "tag" in m.coords:
                    label = f"{m.mark.item()}: {m.tag.item()}"
                else:
                    label = str(m.mark.item())
                if m.valid.item():
                    valid_x += contour_x
                    valid_y += contour_y
                    valid_labels += [label] * len(contour_x)
                else:
                    invalid_x += contour_x
                    invalid_y += contour_y
                    invalid_labels += [label] * len(contour_y)

            fig.add_trace(
                go.Scatter(
                    x=valid_x,
                    y=valid_y,
                    mode="lines",
                    hovertemplate="%{text}<extra></extra>",
                    text=valid_labels,
                    showlegend=False,
                    line_color="green",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=invalid_x,
                    y=invalid_y,
                    mode="lines",
                    hovertemplate="%{text}<extra></extra>",
                    text=invalid_labels,
                    showlegend=False,
                    line_color="red",
                )
            )
        return fig.data

    fig = ndplot(xp, imfunc, animation_frame=animation_frame, facet_col=facet_col, **kwargs)
    fig.update_layout(width=800, height=800, dragmode="pan")
    return fig


def get_contours(roi):
    contours, _ = cv.findContours(
        roi.fg.to_numpy().astype("uint8"),
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )
    # Remove the extra dimension inserted by opencv.
    contours = [c[:, 0] for c in contours]
    # Close the curves.
    contours = [np.append(c, [c[0]], axis=0) for c in contours]
    return contours
