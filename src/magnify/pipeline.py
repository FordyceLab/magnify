from collections.abc import Sequence

import numpy as np
import numpy.ma as ma
import tifffile
import tqdm

from magnify import utils
import magnify


def pipe(
    tile_paths: Sequence,
    times: list[int],
    channels: list[str],
    search_on="egfp",
    drop_images=False,
    progress_bar=False,
    **kwargs
):
    find_kwargs = utils.valid_kwargs(kwargs, magnify.find_buttons)
    segment_kwargs = utils.valid_kwargs(kwargs, magnify.segment_buttons)
    assay = magnify.Assay()
    assay.times = times
    assay.channels = channels
    assays = []
    for time_paths in tqdm.tqdm(tile_paths, disable=not progress_bar):
        images = []
        for channel_paths in time_paths:
            tiles = []
            for row_paths in channel_paths:
                tiles.append([])
                for tile in row_paths:
                    tiles[-1].append(tifffile.imread(tile))
            tiles = np.array(tiles)
            tiles = np.flip(tiles, axis=1)
            images.append(magnify.overlap_stitch(tiles))

        images = np.array(images)
        a = magnify.find_buttons(images, channels, search_on=search_on, **find_kwargs)
        a = magnify.segment_buttons(a, search_on=search_on, **segment_kwargs)
        assays.append(a)

    if not drop_images:
        assay.images = np.concatenate([a.images[np.newaxis] for a in assays])
    assay.regions = np.concatenate([a.regions[np.newaxis] for a in assays])
    assay.offsets = np.concatenate([a.offsets[np.newaxis] for a in assays])
    assay.centers = np.concatenate([a.centers[np.newaxis] for a in assays])
    assay.fg = ma.array(
        assay.regions,
        mask=np.concatenate([ma.getmask(a.fg)[np.newaxis] for a in assays]),
        copy=False,
    )
    assay.bg = ma.array(
        assay.regions,
        mask=np.concatenate([ma.getmask(a.bg)[np.newaxis] for a in assays]),
        copy=False,
    )
    assay.valid = np.concatenate([a.valid[np.newaxis] for a in assays])

    return assay
