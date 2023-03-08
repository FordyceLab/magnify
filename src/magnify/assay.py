from __future__ import annotations
from typing import Sequence

import numpy as np


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
        fg_mask: np.ndarray | None = None,
        bg_mask: np.ndarray | None = None,
    ):
        # Assay metadata.
        self.num_marker_dims = num_marker_dims
        self.times = times
        self.channels = channels
        self.search_channel = search_channel
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

        if fg_mask is not None:
            self.fg_mask = fg_mask
        else:
            self.fg_mask = np.zeros(self.regions.shape, dtype=bool)

        if bg_mask is not None:
            self.bg_mask = bg_mask
        else:
            self.bg_mask = np.zeros(self.regions.shape, dtype=bool)

    @property
    def dims(self) -> str:
        if self.regions:
            "IJTCYX"
        else:
            "ITCYX"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.names.shape + self.images.shape[1:]

    def intensities(
        self,
        time: int,
        channel: int | str,
    ) -> np.ndarray:
        """Return the intensities of each item in the given channel."""
        pass

    @property
    def median(self, t: int, c: int | str, i: int | str, j: int) -> np.ndarray:
        """The median value for the given channel and time."""
        pass

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
            fg_mask=np.concatenate([a.fg_mask for a in assays], axis=-4),
            bg_mask=np.concatenate([a.bg_mask for a in assays], axis=-4),
        )
