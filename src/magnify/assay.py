from __future__ import annotations
from typing import Sequence

import numpy as np
import numpy.ma as ma


class Assay:
    def __init__(
        self,
        num_marker_dims: int,
        times: np.ndarray,
        channels: np.ndarray,
        search_channel: str,
        images: np.ndarray | None = None,
        names: np.ndarray | None = None,
        valid: np.ndarray | None = None,
        centers: np.ndarray | None = None,
        regions: np.ndarray | None = None,
        fg: np.ndarray | None = None,
        bg: np.ndarray | None = None,
    ):
        # Assay metadata.
        self.num_marker_dims = num_marker_dims
        self.times = times
        self.channels = channels
        self.search_channel = search_channel
        self.images_mmaped = False
        self.regions_mmaped = False
        if names is not None:
            self.names = names
        else:
            self.names = np.empty((0,) * self.num_marker_dims)

        # Raw image data.
        if images is not None:
            self.images = images
        else:
            self.images = np.empty((len(self.times), len(self.channels), 0, 0))
        assert self.images.ndim >= 4 and self.images.ndim <= 6

        # Marker data.
        if valid is not None:
            self.valid = valid
        else:
            self.valid = np.ones(
                self.names.shape + (len(self.times), len(self.channels)), dtype=bool
            )

        if centers is not None:
            self.centers = centers
        else:
            self.centers = np.zeros(self.names.shape + (len(self.times), 2))

        if regions is not None:
            self.regions = regions
        else:
            self.regions = np.empty(self.names.shape + (len(self.times), len(self.channels), 0, 0))

        if fg is not None:
            self.fg = fg
        else:
            self.fg = np.zeros(self.regions.shape, dtype=bool)

        if bg is not None:
            self.bg = bg
        else:
            self.bg = np.zeros(self.regions.shape, dtype=bool)

        self.intensities = np.empty(self.regions.shape[-2:])

    def drop_images(self) -> None:
        """Drop the images from the assay."""
        self.images = np.empty((len(self.times), len(self.channels), 0, 0))

    def drop_regions(self) -> None:
        """Drop the regions from the assay."""
        self.regions = np.empty(self.names.shape + (len(self.times), len(self.channels), 0, 0))

    @staticmethod
    def from_assays(assays: Sequence[Assay]) -> Assay:
        """Concatenate the given assays into a single assay along the time dimension."""
        return Assay(
            num_marker_dims=assays[0].num_marker_dims,
            times=np.concatenate([a.times for a in assays]),
            channels=assays[0].channels,
            search_channel=assays[0].search_channel,
            images=np.concatenate([a.images for a in assays], axis=0),
            names=assays[0].names,
            valid=np.concatenate([a.valid for a in assays], axis=-2),
            centers=np.concatenate([a.centers for a in assays], axis=-2),
            regions=np.concatenate([a.regions for a in assays], axis=-4),
            fg=ma.concatenate([a.fg for a in assays], axis=-4),
            bg=ma.concatenate([a.bg for a in assays], axis=-4),
        )
