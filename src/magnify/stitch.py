from __future__ import annotations

import xarray as xr

from magnify.registry import components


class Stitcher:
    def __init__(self, overlap: int = 102):
        self.overlap = overlap

    def __call__(self, assay: xr.Dataset) -> xr.Dataset:
        if "tile_row" not in assay.dims and "tile_col" not in assay.dims:
            return assay
        elif "tile_row" not in assay.dims:
            assay = assay.expand_dims("tile_row", 2)
        elif "tile_col" not in assay.dims:
            assay = assay.expand_dims("tile_col", 3)

        tiles = assay.image[..., : -self.overlap, : -self.overlap]
        # Move the time and channel axes last so we can focus on joining images.
        tiles = tiles.transpose("tile_row", "tile_col", "im_row", "im_col", "channel", "time")
        tiles = xr.concat(tiles, dim="im_row")
        images = xr.concat(tiles, dim="im_col")
        # Move the time and channel axes back to the front.
        images = images.transpose("channel", "time", "im_row", "im_col")
        assay = xr.Dataset(
            {"image": images},
            coords=assay.coords,
            attrs={"search_channel": assay.search_channel},
        )
        return assay

    @components.register("stitcher")
    def make():
        return Stitcher()
