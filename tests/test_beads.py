import numpy as np
import pytest
import xarray as xr

import magnify as mg
from magnify.utils import filled_circle_points


def draw_beads(shape, bead_positions, bead_diameters=20, value=1000):
    # Ensure bead_positions is a 2d numpy array.
    bead_positions = np.array(bead_positions)
    if bead_positions.ndim == 1:
        bead_positions = bead_positions[np.newaxis, :]

    if isinstance(bead_diameters, int):
        bead_diameters = np.full(len(bead_positions), bead_diameters, dtype=np.int32)
    bead_diameters = np.array(bead_diameters)

    bead_radii = bead_diameters // 2
    img = np.zeros(shape, dtype=np.uint16)
    for i in range(len(bead_positions)):
        circle = filled_circle_points(bead_radii[i]) + bead_positions[i]
        img[circle[:, 0], circle[:, 1]] = value
    return img


@pytest.fixture
def bead_single():
    return xr.DataArray(data=draw_beads((1024, 1024), [512, 512]), dims=("y", "x"))


def test_bead_single(bead_single):
    xp = mg.beads(
        data=bead_single,
        min_bead_diameter=16,
        max_bead_diameter=24,
        overlap=0,
        num_iter=100,
    )

    assert xp.roi.sizes["mark"] == 1
    radius = 10
    assert 0.95 * radius < np.sqrt(xp.fg.sum().values.item() / np.pi) < 1.05 * radius
    assert 0.95 * 100 < xp.x.squeeze().values.item() < 1.05 * 100
