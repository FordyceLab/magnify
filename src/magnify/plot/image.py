from __future__ import annotations

import dask.array as da
import napari
import numpy as np
import xarray as xr

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
            roi, channel_axis=xp.roi.dims.index("channel") + 1, name=xp.roi.channel.to_numpy()
        )[0]
    else:
        viewer = napari.imshow(arr)[0]

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

    # Make sure dimension sliders get initialized to be 0.
    viewer.dims.current_step = (0,) * len(img.shape)

    return viewer
