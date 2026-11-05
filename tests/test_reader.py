import pytest

from magnify.reader import Reader


def test_missing_pattern_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(Reader()(str(tmp_path / "nothing*.zarr")))


def test_non_zarr_match_raises_helpful_error(tmp_path):
    (tmp_path / "notzarr").mkdir()
    with pytest.raises(ValueError, match="as a zarr group"):
        list(Reader()(str(tmp_path / "notzarr")))
