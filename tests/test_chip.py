import numpy as np
import pytest
import xarray as xr

import magnify as mg
from magnify.utils import filled_circle_points


def draw_chip(shape, button_diameter=20, row_dist=100, col_dist=100, value=1000, blanks=None):
    """Draw a microfluidic chip with buttons arranged in a grid.

    Args:
        shape: (num_rows, num_cols) of button grid.
        button_diameter: Diameter of each button in pixels.
        row_dist: Distance between button rows in pixels.
        col_dist: Distance between button columns in pixels.
        value: Intensity value for buttons.
        blanks: List of (row, col) tuples for blank positions (no button drawn).
    """
    button_radius = button_diameter // 2
    chip = np.zeros(((shape[0] + 1) * row_dist, (shape[1] + 1) * col_dist), dtype=np.uint16)
    circle = filled_circle_points(button_radius)

    blanks = blanks or []
    blank_set = set(blanks)

    for i in range(shape[0]):
        row_pos = (i + 1) * row_dist
        for j in range(shape[1]):
            if (i, j) in blank_set:
                continue
            col_pos = (j + 1) * col_dist
            chip[circle[:, 0] + row_pos, circle[:, 1] + col_pos] = value
    return chip


@pytest.fixture
def chip_1x1():
    """Minimal 1x1 chip. Used by multiple tests."""
    return xr.DataArray(data=draw_chip((1, 1), 20), dims=("y", "x"))


@pytest.fixture
def chip_2x2():
    """Small 2x2 chip. Used by multiple tests."""
    return xr.DataArray(data=draw_chip((2, 2), 20), dims=("y", "x"))


# =============================================================================
# Tests - Basic Functionality
# =============================================================================


def test_one_by_one_chip(chip_1x1):
    """Test minimal 1x1 chip detection."""
    xp = mg.microfluidic_chip(
        data=chip_1x1,
        shape=(1, 1),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=100,
    )
    assert isinstance(xp, xr.Dataset)
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)

    assert xp.roi.sizes["mark_row"] == 1
    assert xp.roi.sizes["mark_col"] == 1
    radius = 10
    assert 0.95 * radius < np.sqrt(xp.fg.sum().values.item() / np.pi) < 1.05 * radius
    assert 0.95 * 100 < xp.x.squeeze().values.item() < 1.05 * 100


def test_float_chip(chip_1x1):
    """Test that float input is handled correctly."""
    float_chip = chip_1x1.astype(np.float32)
    xp = mg.microfluidic_chip(
        data=float_chip,
        shape=(1, 1),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=100,
    )
    assert isinstance(xp, xr.Dataset)
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)

    assert xp.roi.sizes["mark_row"] == 1
    assert xp.roi.sizes["mark_col"] == 1
    radius = 10
    assert 0.9 * radius < np.sqrt(xp.fg.sum().values.item() / np.pi) < 1.10 * radius
    assert 0.95 * 100 < xp.x.squeeze().values.item() < 1.05 * 100


def test_ten_by_ten_chip():
    """Test standard 10x10 chip detection."""
    data = xr.DataArray(data=draw_chip((10, 10), 20), dims=("y", "x"))

    xp = mg.microfluidic_chip(
        data=data,
        shape=(10, 10),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=10000,
    )
    assert isinstance(xp, xr.Dataset)
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)

    assert xp.roi.sizes["mark_row"] == 10
    assert xp.roi.sizes["mark_col"] == 10
    radius = 10
    radii = np.sqrt(xp.fg.sum(["roi_x", "roi_y"]).to_numpy() / np.pi)
    assert 0.9 * radius < radii.min()
    assert radii.max() < 1.1 * radius
    assert 0.95 * 100 < xp.x[0, 0].values.item() < 1.05 * 100
    assert 0.95 * 100 < xp.y[0, 0].values.item() < 1.05 * 100

    # Check specific positions in grid.
    assert 395 < xp.x[4, 3].values.item() < 405
    assert 495 < xp.y[4, 3].values.item() < 505


# =============================================================================
# Tests - Rectangular Chips
# =============================================================================


def test_rectangular_chip_3x5():
    """Test rectangular chip with more columns than rows."""
    data = xr.DataArray(data=draw_chip((3, 5), 20), dims=("y", "x"))

    xp = mg.microfluidic_chip(
        data=data,
        shape=(3, 5),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=5000,
    )
    assert isinstance(xp, xr.Dataset)
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)

    assert xp.roi.sizes["mark_row"] == 3
    assert xp.roi.sizes["mark_col"] == 5

    # Check corner positions.
    assert 95 < xp.x[0, 0].values.item() < 105
    assert 95 < xp.y[0, 0].values.item() < 105
    assert 495 < xp.x[0, 4].values.item() < 505  # Last column
    assert 295 < xp.y[2, 0].values.item() < 305  # Last row


def test_rectangular_chip_5x3():
    """Test rectangular chip with more rows than columns."""
    data = xr.DataArray(data=draw_chip((5, 3), 20), dims=("y", "x"))

    xp = mg.microfluidic_chip(
        data=data,
        shape=(5, 3),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=5000,
    )
    assert isinstance(xp, xr.Dataset)
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)

    assert xp.roi.sizes["mark_row"] == 5
    assert xp.roi.sizes["mark_col"] == 3

    # Check corner positions.
    assert 95 < xp.x[0, 0].values.item() < 105
    assert 95 < xp.y[0, 0].values.item() < 105
    assert 295 < xp.x[0, 2].values.item() < 305  # Last column
    assert 495 < xp.y[4, 0].values.item() < 505  # Last row


# =============================================================================
# Tests - Different Button Sizes and Spacing
# =============================================================================


def test_large_buttons():
    """Test detection of larger buttons."""
    data = xr.DataArray(
        data=draw_chip((4, 4), button_diameter=40, row_dist=150, col_dist=150), dims=("y", "x")
    )

    xp = mg.microfluidic_chip(
        data=data,
        shape=(4, 4),
        min_button_diameter=30,
        max_button_diameter=50,
        chamber_diameter=100,
        overlap=0,
        row_dist=150,
        col_dist=150,
        num_iter=5000,
    )
    assert isinstance(xp, xr.Dataset)
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)

    assert xp.roi.sizes["mark_row"] == 4
    assert xp.roi.sizes["mark_col"] == 4

    # Check that detected buttons are larger.
    radius = 20
    radii = np.sqrt(xp.fg.sum(["roi_x", "roi_y"]).to_numpy() / np.pi)
    assert 0.85 * radius < radii.min()
    assert radii.max() < 1.15 * radius


def test_rectangular_spacing():
    """Test chip with different row and column spacing."""
    data = xr.DataArray(data=draw_chip((4, 4), 20, row_dist=80, col_dist=120), dims=("y", "x"))

    xp = mg.microfluidic_chip(
        data=data,
        shape=(4, 4),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=80,
        col_dist=120,
        num_iter=5000,
    )
    assert isinstance(xp, xr.Dataset)
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)

    assert xp.roi.sizes["mark_row"] == 4
    assert xp.roi.sizes["mark_col"] == 4

    # Verify spacing is correct.
    # Row 0 to Row 1 should be ~80 pixels apart.
    row_diff = xp.y[1, 0].values.item() - xp.y[0, 0].values.item()
    assert 70 < row_diff < 90

    # Col 0 to Col 1 should be ~120 pixels apart.
    col_diff = xp.x[0, 1].values.item() - xp.x[0, 0].values.item()
    assert 110 < col_diff < 130


# =============================================================================
# Tests - Edge Cases
# =============================================================================


def test_2x2_chip(chip_2x2):
    """Test small 2x2 chip."""
    xp = mg.microfluidic_chip(
        data=chip_2x2,
        shape=(2, 2),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=1000,
    )
    assert isinstance(xp, xr.Dataset)
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)

    assert xp.roi.sizes["mark_row"] == 2
    assert xp.roi.sizes["mark_col"] == 2

    # Check all four positions. x increases with col, y increases with row.
    for i in range(2):
        for j in range(2):
            expected_x = (j + 1) * 100  # col_dist = 100
            expected_y = (i + 1) * 100  # row_dist = 100
            assert 0.9 * expected_x < xp.x[i, j].values.item() < 1.1 * expected_x
            assert 0.9 * expected_y < xp.y[i, j].values.item() < 1.1 * expected_y


def test_chip_with_blanks():
    """Test that chip with blank positions still detects non-blank buttons."""
    blanks = [(0, 0), (1, 2), (2, 1), (3, 3)]
    data = xr.DataArray(data=draw_chip((4, 4), 20, blanks=blanks), dims=("y", "x"))

    xp = mg.microfluidic_chip(
        data=data,
        shape=(4, 4),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=5000,
    )
    assert isinstance(xp, xr.Dataset)
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)

    assert xp.roi.sizes["mark_row"] == 4
    assert xp.roi.sizes["mark_col"] == 4

    # All 16 positions exist but some should have smaller foreground (blank).
    areas = xp.fg.sum(["roi_x", "roi_y"]).to_numpy()
    # Most buttons should have reasonable area.
    good_button_count = np.sum(areas > 100)
    assert good_button_count >= 12  # At least 12 of 16 should be detected


# =============================================================================
# Tests - Output Structure
# =============================================================================


def test_chip_output_structure(chip_2x2):
    """Test that output dataset has expected structure."""
    xp = mg.microfluidic_chip(
        data=chip_2x2,
        shape=(2, 2),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=1000,
    )

    assert isinstance(xp, xr.Dataset)

    # Output is unstacked (restore_format unstacks mark -> mark_row, mark_col).
    assert "mark_row" in xp.dims
    assert "mark_col" in xp.dims

    # Check required coordinates exist.
    assert "x" in xp.coords
    assert "y" in xp.coords
    assert "fg" in xp.coords
    assert "bg" in xp.coords
    assert "tag" in xp.coords

    # Check required data variables.
    assert "roi" in xp.data_vars

    # Check roi dimensions.
    assert "roi_x" in xp.dims
    assert "roi_y" in xp.dims


def test_chip_unstacked_structure(chip_2x2):
    """Test that unstacked output has row/col dimensions."""
    xp = mg.microfluidic_chip(
        data=chip_2x2,
        shape=(2, 2),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=1000,
    )
    assert isinstance(xp, xr.Dataset)
    xp = xp.unstack()

    assert "mark_row" in xp.dims
    assert "mark_col" in xp.dims


# =============================================================================
# Tests - Multi-timestep
# =============================================================================


def test_chip_multiple_timesteps():
    """Test chip detection with multiple timesteps."""
    chip_img = draw_chip((3, 3), 20)
    # Stack same image for 3 timesteps.
    data = xr.DataArray(
        data=np.stack([chip_img, chip_img, chip_img]),
        dims=("time", "y", "x"),
        coords={"time": [0, 1, 2]},
    )

    xp = mg.microfluidic_chip(
        data=data,
        shape=(3, 3),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=5000,
    )

    assert isinstance(xp, xr.Dataset)
    # Should have time dimension in output.
    assert xp.sizes["time"] == 3
    # Positions should be consistent across timesteps.
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)

    # Check positions for all buttons across all timesteps.
    for t in range(3):
        for row in range(3):
            for col in range(3):
                expected_x = (col + 1) * 100
                expected_y = (row + 1) * 100
                actual_x = xp.x[row, col, t].values.item()
                actual_y = xp.y[row, col, t].values.item()
                assert 0.9 * expected_x < actual_x < 1.1 * expected_x
                assert 0.9 * expected_y < actual_y < 1.1 * expected_y

    # Verify button sizes are consistent.
    radius = 10
    areas = xp.fg.sum(dim=["roi_x", "roi_y"]).values
    for area in areas.flatten():
        detected_radius = np.sqrt(area / np.pi)
        assert 0.8 * radius < detected_radius < 1.2 * radius


def test_chip_timestep_refinding():
    """Test that search_timestep controls which timesteps are actively searched."""
    chip_img = draw_chip((3, 3), 20)
    # Stack same image for 4 timesteps.
    data = xr.DataArray(
        data=np.stack([chip_img] * 4),
        dims=("time", "y", "x"),
        coords={"time": [0, 1, 2, 3]},
    )

    # Only search on timestep 0, others should copy from it.
    xp = mg.microfluidic_chip(
        data=data,
        shape=(3, 3),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=5000,
        search_timestep=0,
    )

    assert isinstance(xp, xr.Dataset)
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)

    # All timesteps should have the same button positions (copied from t=0).
    x_t0 = xp.x[:, :, 0].values
    y_t0 = xp.y[:, :, 0].values
    for t in range(1, 4):
        x_t = xp.x[:, :, t].values
        y_t = xp.y[:, :, t].values
        np.testing.assert_array_almost_equal(x_t0, x_t)
        np.testing.assert_array_almost_equal(y_t0, y_t)

    # Verify t=0 positions are correct.
    for row in range(3):
        for col in range(3):
            expected_x = (col + 1) * 100
            expected_y = (row + 1) * 100
            assert 0.9 * expected_x < x_t0[row, col] < 1.1 * expected_x
            assert 0.9 * expected_y < y_t0[row, col] < 1.1 * expected_y


def test_chip_multiple_search_timesteps():
    """Test searching on multiple specific timesteps."""
    chip_img = draw_chip((3, 3), 20)
    data = xr.DataArray(
        data=np.stack([chip_img] * 5),
        dims=("time", "y", "x"),
        coords={"time": [0, 1, 2, 3, 4]},
    )

    # Search on timesteps 0 and 2.
    xp = mg.microfluidic_chip(
        data=data,
        shape=(3, 3),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=5000,
        search_timestep=[0, 2],
    )

    assert isinstance(xp, xr.Dataset)
    assert xp.sizes["time"] == 5
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)

    # Verify positions at searched timesteps.
    for t in [0, 2]:
        for row in range(3):
            for col in range(3):
                expected_x = (col + 1) * 100
                actual_x = xp.x[row, col, t].values.item()
                assert 0.9 * expected_x < actual_x < 1.1 * expected_x


def test_chip_refinding_with_shifted_buttons():
    """Test that refinding adapts to button positions that shift between timesteps."""
    # Create two chip images with buttons at different positions.
    # t=0: buttons at normal positions
    # t=1: buttons shifted by 10 pixels in x and y
    chip_t0 = draw_chip((2, 2), 20, row_dist=100, col_dist=100)

    # For t=1, create a shifted version by padding/cropping.
    shift_y, shift_x = 10, 10
    chip_t1 = np.zeros_like(chip_t0)
    chip_t1[shift_y:, shift_x:] = chip_t0[:-shift_y, :-shift_x]

    data = xr.DataArray(
        data=np.stack([chip_t0, chip_t1]),
        dims=("time", "y", "x"),
        coords={"time": [0, 1]},
    )

    # Search on BOTH timesteps - should find different positions.
    xp = mg.microfluidic_chip(
        data=data,
        shape=(2, 2),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=5000,
        search_timestep=[0, 1],
    )

    assert isinstance(xp, xr.Dataset)
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)

    # t=0 should have buttons at normal positions.
    for row in range(2):
        for col in range(2):
            expected_x = (col + 1) * 100
            expected_y = (row + 1) * 100
            actual_x_t0 = xp.x[row, col, 0].values.item()
            actual_y_t0 = xp.y[row, col, 0].values.item()
            assert 0.9 * expected_x < actual_x_t0 < 1.1 * expected_x
            assert 0.9 * expected_y < actual_y_t0 < 1.1 * expected_y

    # t=1 should have buttons shifted by ~10 pixels.
    for row in range(2):
        for col in range(2):
            expected_x = (col + 1) * 100 + shift_x
            expected_y = (row + 1) * 100 + shift_y
            actual_x_t1 = xp.x[row, col, 1].values.item()
            actual_y_t1 = xp.y[row, col, 1].values.item()
            assert 0.85 * expected_x < actual_x_t1 < 1.15 * expected_x
            assert 0.85 * expected_y < actual_y_t1 < 1.15 * expected_y

    # Verify that t=0 and t=1 positions are different.
    x_diff = np.abs(xp.x[:, :, 1].values - xp.x[:, :, 0].values)
    y_diff = np.abs(xp.y[:, :, 1].values - xp.y[:, :, 0].values)
    assert np.mean(x_diff) > 5  # Should differ by roughly the shift amount
    assert np.mean(y_diff) > 5


def test_chip_no_refinding_copies_from_searched():
    """Test that non-searched timesteps copy positions from searched ones, even if buttons moved."""
    # Create two chip images with buttons at different positions.
    chip_t0 = draw_chip((2, 2), 20, row_dist=100, col_dist=100)

    # t=1 has buttons shifted, but we won't search on it.
    shift_y, shift_x = 15, 15
    chip_t1 = np.zeros_like(chip_t0)
    chip_t1[shift_y:, shift_x:] = chip_t0[:-shift_y, :-shift_x]

    data = xr.DataArray(
        data=np.stack([chip_t0, chip_t1]),
        dims=("time", "y", "x"),
        coords={"time": [0, 1]},
    )

    # Only search on t=0 - t=1 should copy positions from t=0.
    xp = mg.microfluidic_chip(
        data=data,
        shape=(2, 2),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=5000,
        search_timestep=0,
    )

    assert isinstance(xp, xr.Dataset)
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)

    # Both timesteps should have the SAME positions (copied from t=0).
    x_t0 = xp.x[:, :, 0].values
    x_t1 = xp.x[:, :, 1].values
    y_t0 = xp.y[:, :, 0].values
    y_t1 = xp.y[:, :, 1].values

    np.testing.assert_array_almost_equal(x_t0, x_t1)
    np.testing.assert_array_almost_equal(y_t0, y_t1)

    # And those positions should be the original (unshifted) positions.
    for row in range(2):
        for col in range(2):
            expected_x = (col + 1) * 100
            expected_y = (row + 1) * 100
            assert 0.9 * expected_x < x_t0[row, col] < 1.1 * expected_x
            assert 0.9 * expected_y < y_t0[row, col] < 1.1 * expected_y


# =============================================================================
# Tests - Multi-channel
# =============================================================================


def test_chip_multichannel():
    """Test chip detection with multiple channels."""
    chip_img = draw_chip((3, 3), 20)
    # Same chip in both channels.
    data = xr.DataArray(
        data=np.stack([chip_img, chip_img]),
        dims=("channel", "y", "x"),
        coords={"channel": ["bf", "gfp"]},
    )

    xp = mg.microfluidic_chip(
        data=data,
        shape=(3, 3),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=5000,
        search_channel="bf",
    )

    assert isinstance(xp, xr.Dataset)
    assert "bf" in xp.channel.values
    assert "gfp" in xp.channel.values

    # Verify positions.
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)
    for row in range(3):
        for col in range(3):
            expected_x = (col + 1) * 100
            expected_y = (row + 1) * 100
            actual_x = xp.x[row, col].values.item()
            actual_y = xp.y[row, col].values.item()
            assert 0.9 * expected_x < actual_x < 1.1 * expected_x
            assert 0.9 * expected_y < actual_y < 1.1 * expected_y


def test_chip_multichannel_search_specific():
    """Test searching on a specific channel in multi-channel input."""
    # Buttons visible in bf, not in gfp.
    chip_img = draw_chip((3, 3), 20)
    empty_img = np.zeros_like(chip_img)

    data = xr.DataArray(
        data=np.stack([chip_img, empty_img]),
        dims=("channel", "y", "x"),
        coords={"channel": ["bf", "gfp"]},
    )

    # Search on bf channel where buttons are visible.
    xp = mg.microfluidic_chip(
        data=data,
        shape=(3, 3),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=5000,
        search_channel="bf",
    )

    assert isinstance(xp, xr.Dataset)
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)

    # Should find buttons based on bf channel at correct positions.
    for row in range(3):
        for col in range(3):
            expected_x = (col + 1) * 100
            expected_y = (row + 1) * 100
            actual_x = xp.x[row, col].values.item()
            actual_y = xp.y[row, col].values.item()
            assert 0.9 * expected_x < actual_x < 1.1 * expected_x
            assert 0.9 * expected_y < actual_y < 1.1 * expected_y

    # Verify button sizes.
    radius = 10
    areas = xp.fg.sum(dim=["roi_x", "roi_y"]).values
    for area in areas.flatten():
        detected_radius = np.sqrt(area / np.pi)
        assert 0.8 * radius < detected_radius < 1.2 * radius


def test_chip_multichannel_multitimestep():
    """Test chip detection with both multiple channels and timesteps."""
    chip_img = draw_chip((2, 2), 20)
    # 2 channels x 3 timesteps.
    data = xr.DataArray(
        data=np.stack([[chip_img] * 3, [chip_img] * 3]),
        dims=("channel", "time", "y", "x"),
        coords={"channel": ["bf", "gfp"], "time": [0, 1, 2]},
    )

    xp = mg.microfluidic_chip(
        data=data,
        shape=(2, 2),
        min_button_diameter=16,
        max_button_diameter=32,
        overlap=0,
        row_dist=100,
        col_dist=100,
        num_iter=5000,
        search_channel="bf",
    )

    assert isinstance(xp, xr.Dataset)
    assert xp.sizes["time"] == 3
    assert xp.sizes["channel"] == 2

    # Verify positions across all timesteps.
    xp = xp.unstack().transpose("mark_row", "mark_col", ...)
    for t in range(3):
        for row in range(2):
            for col in range(2):
                expected_x = (col + 1) * 100
                expected_y = (row + 1) * 100
                actual_x = xp.x[row, col, t].values.item()
                actual_y = xp.y[row, col, t].values.item()
                assert 0.9 * expected_x < actual_x < 1.1 * expected_x
                assert 0.9 * expected_y < actual_y < 1.1 * expected_y
