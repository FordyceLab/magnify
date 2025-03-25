import numpy as np
import pytest
import xarray as xr

import magnify as mg
from magnify.utils import filled_circle_points


def draw_chip(shape, button_radius=10, row_dist=100, col_dist=100, value=1000):
    chip = np.zeros(((shape[0] + 1) * row_dist, (shape[1] + 1) * col_dist), dtype=np.uint16)
    circle = filled_circle_points(button_radius)
    for i in range(shape[0]):
        row_pos = (i + 1) * row_dist
        for j in range(shape[1]):
            col_pos = (j + 1) * col_dist
            chip[circle[:, 0] + row_pos, circle[:, 1] + col_pos] = value
    return chip


@pytest.fixture
def chip_1x1():
    return xr.DataArray(data=draw_chip((1, 1), 10), dims=("y", "x"))


def test_one_by_one_chip(chip_1x1):
    xp = (
        mg.microfluidic_chip(
            data=chip_1x1,
            shape=(1, 1),
            min_button_diameter=16,
            max_button_diameter=32,
            overlap=0,
            row_dist=100,
            col_dist=100,
            num_iter=100,
        )
        .unstack()
        .transpose("mark_row", "mark_col", ...)
    )

    assert xp.roi.sizes["mark_row"] == 1
    assert xp.roi.sizes["mark_col"] == 1
    radius = 10
    assert 0.95 * radius < np.sqrt(xp.fg.sum().values.item() / np.pi) < 1.05 * radius
    assert 0.95 * 100 < xp.x.squeeze().values.item() < 1.05 * 100
