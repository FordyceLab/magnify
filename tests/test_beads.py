import numpy as np
import pytest
import xarray as xr

import magnify as mg
from magnify.utils import filled_circle_points


def draw_beads(shape, bead_positions, bead_diameters=20, value=1000):
    """Draw filled circles at specified positions."""
    bead_positions = np.array(bead_positions)
    if bead_positions.ndim == 1:
        bead_positions = bead_positions[np.newaxis, :]

    if isinstance(bead_diameters, int):
        bead_diameters = np.full(len(bead_positions), bead_diameters, dtype=np.int32)
    bead_diameters = np.array(bead_diameters)

    if isinstance(value, (int, float)):
        values = np.full(len(bead_positions), value)
    else:
        values = np.array(value)

    bead_radii = bead_diameters // 2
    img = np.zeros(shape, dtype=np.uint16)
    for i in range(len(bead_positions)):
        circle = filled_circle_points(bead_radii[i]) + bead_positions[i]
        # Clip to image bounds.
        valid = (
            (circle[:, 0] >= 0)
            & (circle[:, 0] < shape[0])
            & (circle[:, 1] >= 0)
            & (circle[:, 1] < shape[1])
        )
        img[circle[valid, 0], circle[valid, 1]] = values[i]
    return img


@pytest.fixture
def bead_single():
    """Single bead in center of image. Used by multiple tests."""
    return xr.DataArray(data=draw_beads((1024, 1024), [512, 512]), dims=("y", "x"))


# =============================================================================
# Tests
# =============================================================================


def test_bead_single(bead_single):
    """Test detection of a single centered bead."""
    xp = mg.beads(
        data=bead_single,
        min_bead_diameter=16,
        max_bead_diameter=24,
        overlap=0,
        num_iter=100,
    )

    assert isinstance(xp, xr.Dataset)
    assert xp.roi.sizes["mark"] == 1
    radius = 10
    detected_radius = np.sqrt(xp.fg.sum().values.item() / np.pi)
    assert 0.95 * radius < detected_radius < 1.05 * radius
    assert 0.95 * 512 < xp.x.squeeze().values.item() < 1.05 * 512
    assert 0.95 * 512 < xp.y.squeeze().values.item() < 1.05 * 512


def test_beads_multiple():
    """Test detection of multiple beads spread across image."""
    positions = [
        [200, 200],
        [200, 800],
        [512, 512],
        [800, 200],
        [800, 800],
    ]
    data = xr.DataArray(data=draw_beads((1024, 1024), positions), dims=("y", "x"))

    xp = mg.beads(
        data=data,
        min_bead_diameter=16,
        max_bead_diameter=24,
        overlap=0,
        num_iter=10000,
    )

    assert isinstance(xp, xr.Dataset)
    assert xp.roi.sizes["mark"] == 5

    # Check all beads have reasonable foreground areas.
    radius = 10
    areas = xp.fg.sum(dim=["roi_x", "roi_y"]).values
    radii = np.sqrt(areas / np.pi)
    assert np.all(radii > 0.9 * radius)
    assert np.all(radii < 1.1 * radius)


def test_beads_near_edges():
    """Test detection of beads near image boundaries."""
    positions = [
        [50, 512],  # Near top
        [974, 512],  # Near bottom
        [512, 50],  # Near left
        [512, 974],  # Near right
    ]
    data = xr.DataArray(data=draw_beads((1024, 1024), positions), dims=("y", "x"))

    xp = mg.beads(
        data=data,
        min_bead_diameter=16,
        max_bead_diameter=24,
        overlap=0,
        num_iter=10000,
    )

    assert isinstance(xp, xr.Dataset)
    assert xp.roi.sizes["mark"] == 4

    # Verify positions are near edges.
    x_vals = xp.x.squeeze().values
    y_vals = xp.y.squeeze().values

    # At least one bead should be near each edge.
    assert np.any(y_vals < 100)  # Near top
    assert np.any(y_vals > 900)  # Near bottom
    assert np.any(x_vals < 100)  # Near left
    assert np.any(x_vals > 900)  # Near right


def test_beads_varying_sizes():
    """Test detection of beads with different diameters."""
    positions = [
        [300, 300],
        [300, 700],
        [700, 300],
        [700, 700],
    ]
    diameters = [16, 20, 24, 28]
    data = xr.DataArray(
        data=draw_beads((1024, 1024), positions, bead_diameters=diameters), dims=("y", "x")
    )

    xp = mg.beads(
        data=data,
        min_bead_diameter=14,
        max_bead_diameter=32,
        overlap=0,
        num_iter=10000,
    )

    assert isinstance(xp, xr.Dataset)
    assert xp.roi.sizes["mark"] == 4

    # Check that detected areas vary (different sized beads).
    areas = xp.fg.sum(dim=["roi_x", "roi_y"]).values
    assert areas.max() / areas.min() > 1.5  # Sizes should differ noticeably


def test_beads_close_together():
    """Test that close but non-overlapping beads are detected separately."""
    # With diameter 20 (radius 10), beads at distance 40 apart should be separate.
    bead_positions = [
        [500, 500],
        [500, 540],
        [540, 500],
    ]
    data = xr.DataArray(
        data=draw_beads((1024, 1024), bead_positions, bead_diameters=20), dims=("y", "x")
    )

    xp = mg.beads(
        data=data,
        min_bead_diameter=16,
        max_bead_diameter=24,
        overlap=0,
        num_iter=10000,
    )

    assert isinstance(xp, xr.Dataset)
    assert xp.roi.sizes["mark"] == 3

    # Verify they are at distinct positions.
    detected_positions = np.stack([xp.x.squeeze().values, xp.y.squeeze().values], axis=1)
    for i in range(len(detected_positions)):
        for j in range(i + 1, len(detected_positions)):
            dist = np.linalg.norm(detected_positions[i] - detected_positions[j])
            assert dist > 20  # Should be separated


def test_beads_varying_intensity():
    """Test detection of beads with different intensities."""
    positions = [
        [300, 500],
        [500, 500],
        [700, 500],
    ]
    values = [500, 1000, 2000]
    data = xr.DataArray(data=draw_beads((1024, 1024), positions, value=values), dims=("y", "x"))

    xp = mg.beads(
        data=data,
        min_bead_diameter=16,
        max_bead_diameter=24,
        overlap=0,
        num_iter=10000,
    )

    assert isinstance(xp, xr.Dataset)
    assert xp.roi.sizes["mark"] == 3

    # All beads should be detected regardless of intensity.
    radius = 10
    areas = xp.fg.sum(dim=["roi_x", "roi_y"]).values
    radii = np.sqrt(areas / np.pi)
    assert np.all(radii > 0.85 * radius)


def test_empty_image():
    """Test that empty image returns zero beads."""
    data = xr.DataArray(data=np.zeros((512, 512), dtype=np.uint16), dims=("y", "x"))

    xp = mg.beads(
        data=data,
        min_bead_diameter=16,
        max_bead_diameter=24,
        overlap=0,
        num_iter=100,
    )

    assert isinstance(xp, xr.Dataset)
    assert xp.roi.sizes["mark"] == 0


def test_beads_float_input(bead_single):
    """Test that float input is handled correctly."""
    float_data = bead_single.astype(np.float32)
    xp = mg.beads(
        data=float_data,
        min_bead_diameter=16,
        max_bead_diameter=24,
        overlap=0,
        num_iter=100,
    )

    assert isinstance(xp, xr.Dataset)
    assert xp.roi.sizes["mark"] == 1


def test_beads_output_structure(bead_single):
    """Test that output dataset has expected structure."""
    xp = mg.beads(
        data=bead_single,
        min_bead_diameter=16,
        max_bead_diameter=24,
        overlap=0,
        num_iter=100,
    )

    assert isinstance(xp, xr.Dataset)

    # Check required coordinates exist.
    assert "x" in xp.coords
    assert "y" in xp.coords
    assert "fg" in xp.coords
    assert "bg" in xp.coords

    # Check required data variables.
    assert "roi" in xp.data_vars

    # Check dimensions.
    assert "mark" in xp.dims
    assert "roi_x" in xp.dims
    assert "roi_y" in xp.dims


# =============================================================================
# Tests - Multi-channel
# =============================================================================


def test_beads_multichannel_search_single():
    """Test finding beads on a single channel in multi-channel input."""
    # Create beads at same positions in both channels.
    positions = [[300, 300], [700, 700]]
    ch1 = draw_beads((1024, 1024), positions)
    ch2 = draw_beads((1024, 1024), positions)

    data = xr.DataArray(
        data=np.stack([ch1, ch2]),
        dims=("channel", "y", "x"),
        coords={"channel": ["red", "green"]},
    )

    # Search only on "red" channel.
    xp = mg.beads(
        data=data,
        min_bead_diameter=16,
        max_bead_diameter=24,
        overlap=0,
        num_iter=5000,
        search_channel="red",
    )

    assert isinstance(xp, xr.Dataset)
    assert xp.roi.sizes["mark"] == 2
    assert "red" in xp.channel.values
    assert "green" in xp.channel.values

    # Verify positions are near expected locations.
    detected_positions = set()
    for i in range(2):
        y = xp.y[i].values.item()
        x = xp.x[i].values.item()
        detected_positions.add((round(y / 100) * 100, round(x / 100) * 100))
    assert (300, 300) in detected_positions
    assert (700, 700) in detected_positions

    # Verify bead sizes are reasonable (radius ~10).
    radius = 10
    areas = xp.fg.sum(dim=["roi_x", "roi_y"]).values
    for area in areas:
        detected_radius = np.sqrt(area / np.pi)
        assert 0.8 * radius < detected_radius < 1.2 * radius


def test_beads_multichannel_different_beads():
    """Test finding beads that appear in different channels."""
    # Beads in channel 1 only.
    ch1_positions = [[200, 200], [200, 800]]
    # Beads in channel 2 only.
    ch2_positions = [[800, 200], [800, 800]]

    ch1 = draw_beads((1024, 1024), ch1_positions)
    ch2 = draw_beads((1024, 1024), ch2_positions)

    data = xr.DataArray(
        data=np.stack([ch1, ch2]),
        dims=("channel", "y", "x"),
        coords={"channel": ["red", "green"]},
    )

    # Search on both channels - should find all 4 beads.
    xp = mg.beads(
        data=data,
        min_bead_diameter=16,
        max_bead_diameter=24,
        overlap=0,
        num_iter=10000,
        search_channel=["red", "green"],
    )

    assert isinstance(xp, xr.Dataset)
    assert xp.roi.sizes["mark"] == 4

    # Verify all four positions are found.
    detected_positions = set()
    for i in range(4):
        y = xp.y[i].values.item()
        x = xp.x[i].values.item()
        detected_positions.add((round(y / 100) * 100, round(x / 100) * 100))
    assert (200, 200) in detected_positions
    assert (200, 800) in detected_positions
    assert (800, 200) in detected_positions
    assert (800, 800) in detected_positions


def test_beads_multichannel_subset_only():
    """Test that beads only in non-searched channels are not found."""
    # Beads only in green channel.
    ch1 = np.zeros((1024, 1024), dtype=np.uint16)  # Empty red channel
    ch2 = draw_beads((1024, 1024), [[512, 512]])  # Bead in green channel

    data = xr.DataArray(
        data=np.stack([ch1, ch2]),
        dims=("channel", "y", "x"),
        coords={"channel": ["red", "green"]},
    )

    # Search only on red channel - should find no beads.
    xp = mg.beads(
        data=data,
        min_bead_diameter=16,
        max_bead_diameter=24,
        overlap=0,
        num_iter=1000,
        search_channel="red",
    )

    assert isinstance(xp, xr.Dataset)
    assert xp.roi.sizes["mark"] == 0


def test_beads_multichannel_overlapping():
    """Test beads that appear in multiple channels at same location."""
    # Same bead visible in both channels.
    positions = [[512, 512]]
    ch1 = draw_beads((1024, 1024), positions, value=1000)
    ch2 = draw_beads((1024, 1024), positions, value=2000)

    data = xr.DataArray(
        data=np.stack([ch1, ch2]),
        dims=("channel", "y", "x"),
        coords={"channel": ["red", "green"]},
    )

    # Search on both channels - should find only 1 bead (not duplicated).
    xp = mg.beads(
        data=data,
        min_bead_diameter=16,
        max_bead_diameter=24,
        overlap=0,
        num_iter=5000,
        search_channel=["red", "green"],
    )

    assert isinstance(xp, xr.Dataset)
    assert xp.roi.sizes["mark"] == 1

    # Verify position is near expected location.
    y = xp.y[0].values.item()
    x = xp.x[0].values.item()
    assert 480 < y < 544
    assert 480 < x < 544

    # Verify bead size.
    radius = 10
    area = xp.fg.sum(dim=["roi_x", "roi_y"]).values.item()
    detected_radius = np.sqrt(area / np.pi)
    assert 0.8 * radius < detected_radius < 1.2 * radius
