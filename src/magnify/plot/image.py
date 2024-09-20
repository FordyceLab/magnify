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
):
    settings = napari.settings.get_settings()
    settings.appearance.layer_tooltip_visibility = True
    img = xp.image
    if "channel" in img.dims:
        viewer = napari.imshow(
            img, channel_axis=img.dims.index("channel"), name=img.channel.to_numpy()
        )[0]
    else:
        viewer = napari.imshow(img)[0]

    if "roi" in xp:
        roi = xp.roi.compute()
        # Initialize image metadata.
        roi_contours = []
        fg_labels = np.zeros((img.sizes["im_y"], img.sizes["im_x"]), dtype=int)
        for idx, m in roi.groupby("mark"):
            # Get the centers and the bounds of the bounding box.
            top, bottom, left, right = utils.bounding_box(
                m.x.astype(int).item(),
                m.y.astype(int).item(),
                roi.sizes["roi_y"],
                img.shape[-1],
                img.shape[-2],
            )
            # Set the roi bounding box.
            roi_contours.append(np.array([[top, left], [bottom, right]]))
            # Set the foreground label in image coordinates.
            fg = m.fg.to_numpy()
            while fg.ndim > 2:
                # TODO: Handle channels & time dimensions more elegantly.
                fg = fg[0]
            fg_labels[top:bottom, left:right] = (idx + 1) * fg + fg_labels[
                top:bottom, left:right
            ] * (1 - fg)

        props = {"mark": [i for i in range(xp.sizes["mark"])], "tag": list(xp.tag.to_numpy())}
        viewer.add_labels(
            fg_labels, name="fg", properties={k: [None] + v for k, v in props.items()}
        )
        viewer.add_shapes(
            roi_contours,
            shape_type="rectangle",
            name="roi",
            face_color="transparent",
            text={
                "string": "{mark}: {tag}",
                "size": 10,
                "translation": [-roi.sizes["roi_y"] // 2 + 5, 0],
                "visible": False,
            },
            properties=props,
        )

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
