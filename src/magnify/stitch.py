from __future__ import annotations

import numpy as np
import xarray as xr

from magnify.registry import components


class Stitcher:
    def __init__(self, overlap: int = 102):
        self.overlap = overlap

    def __call__(self, assay: xr.Dataset) -> xr.Dataset:
        tiles = assay.tile[..., : -self.overlap, : -self.overlap]
        # Move the time and channel axes last so we can focus on joining images.
        tiles = tiles.transpose("tile_row", "tile_col", "tile_y", "tile_x", "channel", "time")
        tiles = xr.concat(tiles, dim="tile_y")
        images = xr.concat(tiles, dim="tile_x")
        # Change the x and y dimension names to be about the images.
        images = images.rename(tile_y="im_y", tile_x="im_x")
        # Move the time and channel axes back to the front.
        images = images.transpose("channel", "time", "im_y", "im_x")
        # Add the stitched images to the dataset.
        assay["image"] = images
        return assay

    @components.register("stitch")
    def make(overlap: int = 102):
        return Stitcher(overlap=overlap)
