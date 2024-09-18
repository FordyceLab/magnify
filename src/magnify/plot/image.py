from __future__ import annotations

import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
import napari
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
        contours = get_contours(xp.fg)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([c[:, 0] for c in contours]),
                y=np.concatenate([c[:, 1] for c in contours]),
                mode="lines",
                showlegend=False,
                line_color="green",
            )
        )
        contours = get_contours(xp.bg)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([c[:, 0] for c in contours]),
                y=np.concatenate([c[:, 1] for c in contours]),
                mode="lines",
                showlegend=False,
                line_color="red",
            )
        )
        return fig.data

    fig = ndplot(xp, imfunc, animation_frame=animation_frame, facet_col=facet_col, **kwargs)
    fig.update_layout(width=800, height=800, dragmode="pan")
    return fig


def imshow(
    xp: xr.Dataset,
    contour_type="roi",
    cmap="viridis",
    zmin=None,
    zmax=None,
    **kwargs,
):
    img = xp.image
    if "channel" in img.dims:
        viewer = napari.imshow(img, channel_axis=img.dims.index("channel"), name=img.channel.to_numpy())[0]
    else:
        viewer = napari.imshow(img)[0]

    if "roi" in xp:
        roi = xp.roi.compute()
        # Initialize image metadata.
        valid_x = []
        valid_y = []
        valid_labels = []
        invalid_x = []
        invalid_y = []
        invalid_labels = []
        contours = []
        for idx, m in roi.groupby("mark"):
            # Get the centers and the bounds of the bounding box.
            top, bottom, left, right = utils.bounding_box(
                m.x.item(), m.y.item(), roi.sizes["roi_y"], img.shape[-1], img.shape[-2]
            )
            # Contours are either roi bounding boxes or contours around the foreground.
            if contour_type == "roi":
                contour = [np.array([[top, left], [bottom, right]])]
            elif contour_type == "fg":
                fg = m.fg.to_numpy()
                contour = get_contours(fg[(0,) * (fg.ndim - 2)])
                if len(contour) == 0:
                    continue
                # Adjust contours to be in image coordinates.
                for c in contour:
                    c[:, 0] += top
                    c[:, 1] += left
            contours += contour
        viewer.add_shapes(contours, shape_type="polygon" if contour_type == "fg" else "rectangle")

        """
        # Get the label for the bounding box.
        if "tag" in m.coords:
            label = f"{idx}: {m.tag.item()}"
        else:
            label = str(idx)
        if m.valid.item():
            valid_x += contour_x
            valid_y += contour_y
            valid_labels += [label] * len(contour_x)
        else:
            invalid_x += contour_x
            invalid_y += contour_y
            invalid_labels += [label] * len(contour_y)
        """

        """
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
        """
    return viewer


def get_contours(fg):
    contours, _ = cv.findContours(
        fg.astype("uint8"),
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )
    # Remove the extra dimension inserted by opencv and swap coordinates to match
    # napari's expected input.
    contours = [c[:, 0, ::-1].astype(float) for c in contours]
    return contours
