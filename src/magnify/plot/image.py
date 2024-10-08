from __future__ import annotations

import napari
import napari.settings
import napari.utils
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
    img = xp.image
    if "channel" in img.dims:
        viewer = napari.imshow(
            img, channel_axis=img.dims.index("channel"), name=img.channel.to_numpy()
        )[0]
    else:
        viewer = napari.imshow(img)[0]

    if "roi" in xp:
        # Initialize image metadata.
        extra_dims = [d for d in xp.roi.dims if d not in ["mark", "roi_y", "roi_x"]]
        extra_dim_shape = [xp.sizes[d] for d in extra_dims]
        if len(extra_dims) > 0:
            roi_stack = xp.roi.stack(extra_dims=extra_dims).compute()
        else:
            roi_stack = xp.expand_dims("extra_dims").compute()

        roi_stack = roi_stack.transpose("mark", "extra_dims", "roi_y", "roi_x")
        fg_labels = np.zeros(
            (roi_stack.sizes["extra_dims"], xp.sizes["im_y"], xp.sizes["im_x"]), dtype=int
        )
        roi_contours = np.zeros(
            (roi_stack.sizes["mark"], roi_stack.sizes["extra_dims"], 4, len(extra_dims) + 2),
            dtype=int,
        )
        for i, mark in enumerate(roi_stack.mark):
            for j, d in enumerate(roi_stack.extra_dims):
                m = roi_stack.sel(mark=mark, extra_dims=d)
                # Get the centers and the bounds of the bounding box.
                top, bottom, left, right = utils.bounding_box(
                    m.x.astype(int).item(),
                    m.y.astype(int).item(),
                    roi_stack.sizes["roi_y"],
                    img.shape[-1],
                    img.shape[-2],
                )
                # Set the roi bounding box.
                roi_contours[i, j, :, :-2] = np.unravel_index(j, extra_dim_shape)
                roi_contours[i, j, :, -2:] = np.array(
                    [[top, left], [top, right], [bottom, right], [bottom, left]], dtype=int
                )
                # Set the foreground label in image coordinates.
                fg = m.fg.to_numpy()
                fg_labels[j, top:bottom, left:right] = (i + 1) * fg + fg_labels[
                    j, top:bottom, left:right
                ] * (1 - fg)

        fg_labels = fg_labels.reshape(img.shape)
        roi_contours = roi_contours.reshape(-1, 4, len(extra_dims) + 2)
        props = {"mark": [f"{mark.item()}" for mark in xp.mark], "tag": list(xp.tag.to_numpy())}
        viewer.add_labels(
            fg_labels, name="fg", properties={k: [None] + v for k, v in props.items()}
        )
        props["mark"] = np.repeat(props["mark"], roi_stack.sizes["extra_dims"])
        props["tag"] = np.repeat(props["tag"], roi_stack.sizes["extra_dims"])
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
        )

    # Make sure dimension sliders get initialized to be 0.
    viewer.dims.current_step = (0,) * len(img.shape)

    return viewer
