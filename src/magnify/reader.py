from __future__ import annotations
import datetime
import fnmatch
import glob
import os
import re

import dask
import dask.array as da
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
import tifffile
import xarray as xr

from magnify.assay import Assay
from magnify.pipeline import Pipeline
import magnify.registry as registry


class ChipReader:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        data: ArrayLike | str,
        names: ArrayLike | str,
        search_on: str = "egfp",
        times: Sequence[int] | None = None,
        channels: Sequence[str] | None = None,
    ) -> Assay:
        if isinstance(data, str):
            path_dict = extract_paths(data)
            if len(path_dict) == 0:
                raise FileNotFoundError(f"The pattern {data} did not lead to any files.")
            times, channels, rows, cols = (sorted(set(idx)) for idx in zip(*path_dict.keys()))

        names_array = read_names(names)

        for time in times:
            path_list = [path_dict[key] for key in sorted(path_dict) if key[0] == time]
            tiles = np.stack(tifffile.imread(path_list), axis=0)
            tiles = np.reshape(
                tiles,
                (
                    1,
                    len(channels),
                    len(rows),
                    len(cols),
                    *tiles.shape[1:],
                ),
            )
            yield Assay(
                num_marker_dims=2,
                times=np.array([time]),
                channels=np.array(channels),
                search_channel=search_on,
                images=tiles,
                names=names_array,
            )

    @registry.readers.register("chip_reader")
    def make():
        return ChipReader()


class BeadReader:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        data: ArrayLike | str,
        names: ArrayLike | str,
        search_on: str = "open",
        times: Sequence[int] | None = None,
        channels: Sequence[str] | None = None,
    ) -> Assay:
        if isinstance(data, str):
            data = [data]

        for d in data:
            path_dict = extract_paths(d)
            if len(path_dict) == 0:
                raise FileNotFoundError(f"The pattern {d} did not lead to any files.")
            temp_times, temp_channels, rows, cols = (
                sorted(set(idx)) for idx in zip(*path_dict.keys())
            )
            if times is None:
                times = temp_times
            if channels is None:
                channels = temp_channels
            with tifffile.TiffFile(
                path_dict[temp_times[0], temp_channels[0], rows[0], cols[0]]
            ) as tif:
                dims = tif.series[0].axes
                if "T" in dims:
                    raise ValueError("ome.tiff files with a time dimension are not yet supported.")
                if times[0] == -1:
                    times = [0]

                if channels[0] == "":
                    channels = tif.micromanager_metadata["Summary"]["ChNames"]

                if rows[0] != -1 or cols[0] != -1:
                    raise ValueError("Tiled images not yet supported.")

                if "Z" in dims:
                    raise ValueError("ome.tiff files with a Z dimension are not yet supported.")
                if "X" not in dims or "Y" not in dims:
                    raise ValueError("ome.tiff files must contain an X and Y dimension.")

                dtype = tif.series[0].dtype
                shape = tif.series[0].shape

            imread = dask.delayed(lambda x: tifffile.TiffFile(x).asarray(), pure=True)
            images = []
            for time in times:
                path = [path_dict[key] for key in sorted(path_dict) if key[0] == time][0]
                images.append(da.from_delayed(imread(path), dtype=dtype, shape=shape))
            if len(shape) == 2:
                images = da.stack(images, axis=0)[np.newaxis]
            else:
                images = da.stack(images, axis=1)
            yield xr.Dataset(
                {"image": (["channel", "time", "im_row", "im_col"], images)},
                coords={"channel": channels, "time": times},
                attrs={"search_channel": search_on},
            )

    @registry.readers.register("bead_reader")
    def make():
        return BeadReader()


def read_names(path):
    df = pd.read_csv(path)
    df["Indices"] = df["Indices"].apply(
        lambda s: [int(x) for x in re.sub(r"[\(\)]", "", s).split(",")]
    )
    # Zero-index the indices.
    cols, rows = np.array(df["Indices"].to_list()).T - 1
    names = df["MutantID"].to_numpy(dtype=str, na_value="")
    names_array = np.empty((max(rows) + 1, max(cols) + 1), dtype=names.dtype)
    names_array[rows, cols] = names
    return names_array


def extract_paths(pattern) -> dict[tuple[int, str, int, int], str]:
    pattern = os.path.expanduser(pattern)

    # Filter out special signifiers used for pattern matching.
    glob_path = re.sub(r"\(time\s*\|?.*?\)", "*", pattern)
    glob_path = glob_path.replace("(channel)", "*")
    glob_path = glob_path.replace("(row)", "*")
    glob_path = glob_path.replace("(col)", "*")

    # Search for files matching the pattern.
    paths = glob.glob(glob_path, recursive=True)

    regex_path = fnmatch.translate(pattern)
    regex_path = re.sub(r"\\\(time\s*\|?.*?\\\)", r"(?P<time>.*?)", regex_path)
    regex_path = regex_path.replace("\\(channel\\)", "(?P<channel>.*?)")
    regex_path = regex_path.replace("\\(row\\)", "(?P<row>.*?)")
    regex_path = regex_path.replace("\\(col\\)", "(?P<col>.*?)")
    regex_path = re.compile(regex_path, re.IGNORECASE)

    path_dict = {}
    for path in paths:
        match = regex_path.fullmatch(path)
        time_search = re.search(r"\(time\s*\|?\s*(.*?)\)", pattern)
        if time_search:
            format_str = time_search.group(1)
            if not format_str:
                format_str = "%Y%m%d-%H%M%S"
            time_str = match.group("time")
            time = datetime.datetime.strptime(time_str, format_str).timestamp()
        else:
            time = 0

        if "(channel)" in pattern:
            channel = match.group("channel")
        else:
            channel = ""

        if "(row)" in pattern:
            row = int(match.group("row"))
        else:
            row = -1

        if "(col)" in pattern:
            col = int(match.group("col"))
        else:
            col = -1

        idx = (time, channel, row, col)
        if idx not in path_dict:
            path_dict[idx] = path
        else:
            raise ValueError(f"{path} and {path_dict[idx]} map to the same index.")

    return path_dict
