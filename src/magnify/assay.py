import numpy as np


class Assay:
    # The type of assay that we imaged can be "bead" or "chip".
    # In a chip assay the items we are imaging are buttons, whereas
    # in a bead assay we are imaging beads.
    type: str
    # The names of each item. Names do not have to be unique.
    # This is a num_row x num_col array for chip assays,
    # and a num_item array for bead assays.
    names: np.ndarray
    # The time in seconds at which each image was taken.
    times: np.ndarray
    # The channels each image was acquired in.
    channels: np.ndarray

    # The images acquired for this assay.
    # This is a num_times x num_channels x image_height x image_width array.
    images: np.ndarray
    # The center of each item in row-col coordinates.
    # This is a num_times x num_channels x num_items x 2 array.
    centers: np.ndarray
    # Subsets of the images array that contains items
    regions: np.ndarray
    # The row/column offsets of each region in the 
    offsets: np.ndarray

    fg: np.ndarray
    bg: np.ndarray
    valid: np.ndarray
