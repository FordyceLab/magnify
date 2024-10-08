from __future__ import annotations

import xarray as xr

from magnify.registry import components


class Stitcher:
    def __init__(self, overlap: int = 102):
        self.overlap = overlap

    def __call__(self, assay: xr.Dataset) -> xr.Dataset:
        tiles = assay.tile[
            ..., : assay.tile.shape[-2] - self.overlap, : assay.tile.shape[-1] - self.overlap
        ]
        # Move the time and channel axes last so we can focus on joining images.
        tiles = tiles.transpose("tile_row", "tile_col", "tile_y", "tile_x", "channel", "time")
        tiles = xr.concat(tiles, dim="tile_y")
        images = xr.concat(tiles, dim="tile_x")
        # Change the x and y dimension names to be about the images.
        images = images.rename(tile_y="im_y", tile_x="im_x")
        # Move the time and channel axes back to the front.
        images = images.transpose("channel", "time", "im_y", "im_x")

        # Rechunk the array so each chunk is a single tile and cache the intermediate results.
        assay["image"] = images.chunk(
            (1, 1, assay.sizes["tile_y"], assay.sizes["tile_x"])
        ).mg.cache()

        return assay

    @components.register("stitch")
    def make(overlap: int = 102):
        return Stitcher(overlap=overlap)
