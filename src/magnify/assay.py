import numpy as np


class Marker:
    def __init__(self, name, x, y, region, fg_mask, bg_mask):
        



class Assay:
    def __init__(self, images: ArrayLike, names: ArrayLike, times: ArrayLike, channels: ArrayLike, region_height=60, region_width=60):
        self.images: np.ndarray | None = np.asarray(images)
        self.names: np.ndarray = np.asarray(names)
        self.times: np.ndarray = np.asarray(times)
        self.channels: np.ndarray = np.asarray(channels)
        self.markers: list[Marker]
        assert self.images.ndim >= 4 and self.images.ndim <= 6

        self.valid: np.ndarray | None = None
        self.centers: np.ndarray | None = None

        self.offsets: np.ndarray | None = None
        self.regions: np.ndarray | None = None
        self.fg_mask: np.ndarray | None = None
        self.bg_mask: np.ndarray | None = None

        self.intensity: 

    @property
    def dims(self) -> str:
        if self.regions :
            "IJTCYX"
        else:
            "ITCYX"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.names.shape + self.images.shape[1:]

    def intensities(self, time: int, channel: int | str, ) -> np.ndarray:
        """Return the intensities of each item in the given channel."""
        pass

    @property
    def median(self, t: int, c: int | str, i: int | str, j: int) -> np.ndarray:
        """The median value for the given channel and time."""
        pass

def concatenate(assays: Sequence[Assay], axis: int | str = 0) -> Assay:
    """Concatenate the given assays into a single assay.
    
    # The assays must have the same type, channels, and names.
    # The times and images arrays will be concatenated.
    # The centers and regions arrays will be concatenated,
    # and the offsets will be adjusted to account for the
    # concatenation.
    """
    pass
