from __future__ import annotations

import numpy as np

from magnify.assay import Assay
from magnify.registry import components


class Stitcher:
    def __init__(self, overlap: int = 102):
        self.overlap = overlap

    def __call__(self, assay: Assay) -> Assay:
        tiles = assay.images[..., : -self.overlap, : -self.overlap]
        # Move the time and channel axes last so we can focus on joining images.
        tiles = np.transpose(tiles, axes=(2, 3, 4, 5, 0, 1))
        # Concatenate rows, the axis corresponding to image rows is one less than usual since
        # numpy only starts counting from the axis at index 1 in the input array.
        tiles = np.concatenate(tiles, axis=1)
        # Concatenate columns. Now the image column axis is two less than usual since we've
        # also removed one axis with our last concatenation.
        tiles = np.concatenate(tiles, axis=1)
        # Move the time and channel axes back to the front.
        assay.images = np.transpose(tiles, axes=(2, 3, 0, 1))
        return assay

    @components.register("stitcher")
    def make():
        return Stitcher()
