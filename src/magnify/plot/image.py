from __future__ import annotations

import napari
import napari.settings
import napari.utils
import numba
import numpy as np
import xarray as xr
from numba import prange

import magnify.utils as utils


def roishow(xp: xr.Dataset):
    # TODO: This entire section doesn't handle images with time dimensions correctly.
    tags, counts = np.unique(xp.tag.to_numpy(), return_counts=True)
    roi = np.zeros((counts.max(), len(tags)) + xp.roi.isel(mark=0).shape)
    fg = np.zeros((counts.max(), len(tags)) + xp.roi.isel(mark=0, channel=0).shape, dtype=bool)
    bg = np.zeros_like(fg)
    for i, (tag, group) in enumerate(xp.roi.groupby("tag")):
        roi[: group.sizes["mark"], i] = group
        fg[: group.sizes["mark"], i] = group.fg.isel(channel=0)
        bg[: group.sizes["mark"], i] = group.bg.isel(channel=0)

    if "channel" in xp.roi.dims:
        viewer = napari.imshow(
            roi,
            channel_axis=xp.roi.dims.index("channel") + 1,
            name=xp.roi.channel.to_numpy(),
        )[0]
    else:
        viewer = napari.imshow(roi)[0]

    viewer.add_labels(
        bg,
        name="bg",
        colormap=napari.utils.CyclicLabelColormap([(0, 0, 0, 0), (1, 0, 0, 0.7)]),
    )
    viewer.add_labels(
        fg,
        name="fg",
        colormap=napari.utils.CyclicLabelColormap([(0, 0, 0, 0), (0, 1.0, 0, 0.7)]),
    )

    viewer.dims.axis_labels = ("mark", "tag", "y", "x")
    # Make sure dimension sliders get initialized to be 0.
    viewer.dims.current_step = (0,) * len(roi.shape)

    return viewer


def imshow(xp: xr.Dataset):
    settings = napari.settings.get_settings()
    settings.appearance.layer_tooltip_visibility = True
    if "mark_row" in xp.dims and "mark_col" in xp.dims and "mark" not in xp.dims:
        xp = xp.stack(mark=("mark_row", "mark_col"))
    xp = xp.transpose(..., "im_y", "im_x")
    img = xp.image

    multiscale = [img]
    while multiscale[-1].sizes["im_x"] * multiscale[-1].sizes["im_y"] > 512**2:
        multiscale.append(multiscale[-1][..., ::2, ::2])

    if "channel" in img.dims:
        viewer = napari.imshow(
            multiscale,
            multiscale=True,
            channel_axis=img.dims.index("channel"),
            name=img.channel.to_numpy(),
        )[0]
    else:
        viewer = napari.imshow(multiscale, multiscale=True, name="image")[0]

    if "roi" in xp:
        # Initialize image metadata.
        extra_dims = [d for d in xp.fg.dims if d not in ["mark", "roi_y", "roi_x"]]
        extra_dim_shape = [xp.sizes[d] for d in extra_dims]
        if len(extra_dims) > 0:
            fg_stack = xp.fg.stack(extra_dims=extra_dims).compute()
            xs = xp.x.stack(extra_dims=extra_dims)
            ys = xp.y.stack(extra_dims=extra_dims)
        else:
            fg_stack = xp.fg.expand_dims("extra_dims").compute()
            xs = xp.x.expand_dims("extra_dims")
            ys = xp.y.expand_dims("extra_dims")

        fg_stack = fg_stack.transpose("mark", "extra_dims", "roi_y", "roi_x")
        xs = xs.transpose("mark", "extra_dims").to_numpy().astype(int)
        ys = ys.transpose("mark", "extra_dims").to_numpy().astype(int)

        roi_contours = np.zeros(
            (
                fg_stack.sizes["mark"],
                fg_stack.sizes["extra_dims"],
                4,
                len(extra_dims) + 2,
            ),
            dtype=int,
        )
        tblr = np.zeros((fg_stack.sizes["mark"], fg_stack.sizes["extra_dims"], 4), dtype=int)
        for i in range(fg_stack.sizes["mark"]):
            for j in range(fg_stack.sizes["extra_dims"]):
                # Get the centers and the bounds of the bounding box.
                top, bottom, left, right = utils.bounding_box(
                    xs[i, j],
                    ys[i, j],
                    fg_stack.sizes["roi_y"],
                    img.sizes["im_x"],
                    img.sizes["im_y"],
                )
                tblr[i, j] = top, bottom, left, right
                # Set the roi bounding box.
                roi_contours[i, j, :, :-2] = np.unravel_index(j, extra_dim_shape)
                roi_contours[i, j, :, -2:] = np.array(
                    [[top, left], [top, right], [bottom, right], [bottom, left]],
                    dtype=int,
                )

        # Set the foreground label in image coordinates.
        fg_labels = roi_to_image_labels(fg_stack.to_numpy(), tblr, img.shape[-2:])

        fg_labels = fg_labels.reshape(
            img.isel(channel=0).shape if "channel" in img.dims else img.shape
        )
        roi_contours = roi_contours.reshape(-1, 4, len(extra_dims) + 2)
        props = {
            "mark": [f"{mark.item()}" for mark in xp.mark],
            "tag": list(xp.tag.to_numpy()) if "tag" in xp else [""] * xp.sizes["mark"],
        }
        viewer.add_labels(
            fg_labels, name="fg", properties={k: [None] + v for k, v in props.items()}
        )
        props["mark"] = np.repeat(props["mark"], fg_stack.sizes["extra_dims"])
        props["tag"] = np.repeat(props["tag"], fg_stack.sizes["extra_dims"])
        viewer.add_shapes(
            roi_contours,
            shape_type="rectangle",
            name="roi",
            edge_color="white",
            edge_width=2,
            face_color="transparent",
            text={
                "string": "{mark}: {tag}",
                "size": 10,
                "translation": [0] * len(extra_dims) + [-xp.sizes["roi_y"] // 2 + 5, 0],
                "visible": False,
            },
            properties=props,
            visible=False,
        )
    # Make sure dimension sliders get initialized to be 0.
    viewer.dims.current_step = (0,) * len(img.shape)

    return viewer


@numba.njit(parallel=True)
def roi_to_image_labels(roi_masks, bboxes, img_shape):
    img_labels = np.zeros((roi_masks.shape[1],) + img_shape, dtype=np.int32)
    for i in range(roi_masks.shape[0]):
        for j in prange(roi_masks.shape[1]):
            mask = roi_masks[i, j]
            top, bottom, left, right = bboxes[i, j]
            img_labels[j, top:bottom, left:right] = (i + 1) * mask + img_labels[
                j, top:bottom, left:right
            ] * (1 - mask)

    return img_labels
