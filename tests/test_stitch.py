import numpy as np
import pytest
import xarray as xr

from magnify.stitch import Stitcher


class TestStitcher:
    def test_stitcher_basic(self):
        stitcher = Stitcher(overlap=5)
        tile_data = np.random.rand(1, 1, 2, 3, 40, 40)
        test_dataset = xr.Dataset(
            {
                "tile": xr.DataArray(
                    tile_data,
                    dims=["channel", "time", "tile_row", "tile_col", "tile_y", "tile_x"],
                )
            }
        )
        result = stitcher(test_dataset)
        assert "image" in result.data_vars
        assert result.sizes["im_y"] == 2 * (40 - 5)
        assert result.sizes["im_x"] == 3 * (40 - 5)
        np.testing.assert_array_equal(
            result.image[0, 0, 35:70, 35:70], tile_data[0, 0, 1, 1, 2:37, 2:37]
        )

    def test_stitcher_single_tile(self):
        stitcher = Stitcher(overlap=5)
        tile_data = np.random.rand(1, 1, 1, 1, 30, 30)
        test_dataset = xr.Dataset(
            {
                "tile": xr.DataArray(
                    tile_data,
                    dims=["channel", "time", "tile_row", "tile_col", "tile_y", "tile_x"],
                )
            }
        )

        result = stitcher(test_dataset)

        assert "image" in result.data_vars
        assert result.sizes["im_y"] == 30 - 5
        assert result.sizes["im_x"] == 30 - 5
        np.testing.assert_array_equal(result.image[0, 0], tile_data[0, 0, 0, 0, 2:27, 2:27])

    def test_stitcher_preserves_channels_and_time(self):
        stitcher = Stitcher(overlap=8)

        # Multi-channel, multi-time tile data.
        tile_data = np.random.rand(2, 3, 2, 2, 25, 25)
        test_dataset = xr.Dataset(
            {
                "tile": xr.DataArray(
                    tile_data,
                    dims=["channel", "time", "tile_row", "tile_col", "tile_y", "tile_x"],
                    coords={
                        "channel": ["red", "green"],
                        "time": [0, 1, 2],
                        "tile_row": [0, 1],
                        "tile_col": [0, 1],
                    },
                )
            }
        )

        result = stitcher(test_dataset)

        assert isinstance(result, xr.Dataset)
        assert "image" in result.data_vars

        # Should preserve channel and time dimensions.
        assert "channel" in result.image.dims
        assert "time" in result.image.dims
        assert len(result.channel) == 2
        assert len(result.time) == 3

    def test_stitcher_zero_overlap(self):
        stitcher = Stitcher(overlap=0)
        tile_data = np.random.rand(1, 1, 1, 2, 20, 20)
        test_dataset = xr.Dataset(
            {
                "tile": xr.DataArray(
                    tile_data,
                    dims=["channel", "time", "tile_row", "tile_col", "tile_y", "tile_x"],
                )
            }
        )

        result = stitcher(test_dataset)

        assert "image" in result.data_vars
        assert result.sizes["im_y"] == 20
        assert result.sizes["im_x"] == 40
        np.testing.assert_array_equal(result.image[0, 0, :, :20], tile_data[0, 0, 0, 0])
        np.testing.assert_array_equal(result.image[0, 0, :, 20:], tile_data[0, 0, 0, 1])

    def test_stitcher_invalid_overlap(self):
        with pytest.raises(ValueError):
            Stitcher(overlap=-5)

    def test_stitcher_missing_tile_data(self):
        # Test stitcher behavior when tile data is missing.
        stitcher = Stitcher(overlap=10)

        # Dataset without tile data.
        empty_dataset = xr.Dataset({"other_data": xr.DataArray([1, 2, 3], dims=["x"])})

        with pytest.raises(AttributeError):
            stitcher(empty_dataset)

    def test_stitcher_large_overlap(self):
        stitcher = Stitcher(overlap=100)
        tile_data = np.random.rand(1, 1, 2, 2, 50, 50)
        test_dataset = xr.Dataset(
            {
                "tile": xr.DataArray(
                    tile_data,
                    dims=["channel", "time", "tile_row", "tile_col", "tile_y", "tile_x"],
                )
            }
        )

        with pytest.raises(ValueError):
            stitcher(test_dataset)
