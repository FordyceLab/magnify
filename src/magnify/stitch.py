import xarray as xr

from magnify.registry import components


class Stitcher:
    def __init__(self, overlap: int = 102):
        if overlap < 0:
            raise ValueError("Overlap must be non-negative.")
        self.overlap = overlap

    def __call__(self, assay: xr.Dataset) -> xr.Dataset:
        if "tile" not in assay:
            raise AttributeError("Dataset must contain 'tile' data variable.")

        if self.overlap >= assay.sizes["tile_y"] or self.overlap >= assay.sizes["tile_x"]:
            raise ValueError(
                f"Overlap ({self.overlap}) must be smaller than tile size ({assay.sizes['tile_y']}x{assay.sizes['tile_x']})."
            )

        # Take half of overlap from each edge.
        clip = self.overlap // 2
        # Account for odd overlaps.
        remainder = self.overlap % 2
        tiles = assay.tile[
            ...,
            clip : assay.tile.shape[-2] - clip - remainder,
            clip : assay.tile.shape[-1] - clip - remainder,
        ]

        # Move the time and channel axes last so we can focus on joining images.
        tiles = tiles.transpose("tile_row", "tile_col", "tile_y", "tile_x", "channel", "time")
        tiles = xr.concat(tiles, dim="tile_y", coords="different", compat="equals")
        images = xr.concat(tiles, dim="tile_x", coords="different", compat="equals")
        # Change the x and y dimension names to be about the images.
        images = images.rename(tile_y="im_y", tile_x="im_x")
        # Move the time and channel axes back to the front.
        images = images.transpose("channel", "time", "im_y", "im_x")

        # Rechunk the array so each chunk is a single tile and cache the intermediate results.
        assay["image"] = images.chunk(
            {"channel": 1, "time": 1, "im_y": assay.sizes["tile_y"], "im_x": assay.sizes["tile_x"]}
        )
        assay.mg.cache("image")
        return assay

    @components.register("stitch")
    def make(overlap: int = 102):
        return Stitcher(overlap=overlap)
