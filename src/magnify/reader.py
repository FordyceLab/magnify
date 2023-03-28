from __future__ import annotations
from collections.abc import Iterator
import collections
import datetime
import fnmatch
import functools
import glob
import os
import re

from numpy.typing import ArrayLike
from typing import Iterator
import dask.array as da
import numpy as np
import pandas as pd
import tifffile
import xarray as xr

from magnify.pipeline import Pipeline
import magnify.registry as registry
import magnify.utils as utils


class Reader:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        data: ArrayLike | str,
        names: ArrayLike | str,
        search_on: str = "all",
        times: Sequence[int] | None = None,
        channels: Sequence[str] | None = None,
    ) -> Iterator[xr.Dataset]:
        if isinstance(data, str):
            data = [data]

        for d in data:
            path_dict = extract_paths(d)
            if len(path_dict) == 0:
                raise FileNotFoundError(f"The pattern {d} did not lead to any files.")

            for assay_name, assay_dict in sorted(
                path_dict.items(), key=lambda x: utils.natural_sort_key(x[0])
            ):
                # Use these variables within the loop so we don't affect other assays.
                channel_coords = channels
                time_coords = times

                # Get the indices specified in the path each in sorted order.
                channel_idxs, time_idxs, row_idxs, col_idxs = (
                    sorted(set(idx)) for idx in zip(*assay_dict.keys())
                )

                # Check which dimensions are specified in the path and compute the shape of those dimensions.
                dims_in_path = []
                outer_shape = tuple()
                if channel_idxs[0] != -1:
                    dims_in_path.append("channel")
                    outer_shape += (len(channel_idxs),)
                if time_idxs[0] != -1:
                    dims_in_path.append("time")
                    outer_shape += (len(time_idxs),)
                if row_idxs[0] != -1:
                    dims_in_path.append("tile_row")
                    outer_shape += (len(row_idxs),)
                if col_idxs[0] != -1:
                    dims_in_path.append("tile_col")
                    outer_shape += (len(col_idxs),)

                # If the user didn't specify times or channels use the ones from the path.
                if time_coords is None and "time" in dims_in_path:
                    time_coords = time_idxs
                if channel_coords is None and "channel" in dims_in_path:
                    channel_coords = channel_idxs

                # Read in a single image to get the metadata stored within the file.
                with tifffile.TiffFile(next(iter(assay_dict.values()))) as tif:
                    letter_to_dim = {
                        "C": "channel",
                        "T": "time",
                        "Z": "depth",
                        "Y": "im_row",
                        "X": "im_col",
                    }
                    dims_in_file = [letter_to_dim[c] for c in tif.series[0].axes]

                    if channel_coords is None and "channel" in dims_in_file:
                        channel_coords = tif.micromanager_metadata["Summary"]["ChNames"]

                    if "time" in dims_in_file:
                        raise ValueError("tiff files with a time dimension are not yet supported.")
                    if "depth" in dims_in_file:
                        raise ValueError("tiff files with a Z dimension are not yet supported.")
                    if "im_row" not in dims_in_file or "im_col" not in dims_in_file:
                        raise ValueError("tiff files must contain an X and Y dimension.")

                    dtype = tif.series[0].dtype
                    inner_shape = tif.series[0].shape

                # Check the dimensions specified in the path and inside the tiff file do not overlap.
                if set(dims_in_file).intersection(dims_in_path):
                    raise ValueError(
                        "Dimensions specified in the path names and inside the tiff file overlap."
                    )

                # Setup a dask array to lazyload images.
                filenames = [path for _, path in sorted(assay_dict.items())]

                def read_image(block_id, filenames):
                    block_id = block_id[: len(outer_shape)]
                    idx = np.ravel_multi_index(block_id, outer_shape)
                    with tifffile.TiffFile(filenames[idx]) as tif:
                        return np.expand_dims(tif.asarray(), axis=tuple(range(len(block_id))))

                # Chunk the images so that each chunk represents a single file.
                images = da.map_blocks(
                    functools.partial(read_image, filenames=filenames),
                    dtype=dtype,
                    chunks=((tuple((1,) * size for size in outer_shape) + inner_shape)),
                )

                # Set named coordinates for channels and times if they're available.
                coords = {}
                if channel_coords is not None:
                    coords["channel"] = channel_coords
                if time_coords is not None:
                    coords["time"] = time_coords

                # Put all our data into an xarray dataset.
                assay = xr.Dataset(
                    {"image": (dims_in_path + dims_in_file, images)},
                    coords=coords,
                    attrs={
                        "name": assay_name,
                        "search_channel": search_on,
                    },
                )

                # Make sure the assay always has a time and channel dimension.
                if "channel" not in assay.dims:
                    assay = assay.expand_dims("channel", 0)
                if "time" not in assay.dims:
                    assay = assay.expand_dims("time", 1)

                # Reorder the dimensions so they're always consistent.
                desired_order = []
                for dim in ["channel", "time", "tile_row", "tile_col", "im_row", "im_col"]:
                    if dim in assay.dims:
                        desired_order.append(dim)
                assay.transpose(*desired_order)

                yield assay

    @registry.readers.register("read")
    def make():
        return Reader()


@registry.components.register("read_pinlist")
def make_read_pinlist():
    def read_pinlist(assay, pinlist):
        assay = assay.assign_coords(id=(("marker_row", "marker_col"), read_names(pinlist)))
        return assay

    return read_pinlist


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
    glob_path = pattern.replace("(assay)", "*")
    glob_path = glob_path.replace("(channel)", "*")
    glob_path = re.sub(r"\(time\s*\|?.*?\)", "*", glob_path)
    glob_path = glob_path.replace("(row)", "*")
    glob_path = glob_path.replace("(col)", "*")

    # Search for files matching the pattern.
    paths = glob.glob(glob_path, recursive=True)

    regex_path = fnmatch.translate(pattern)
    regex_path = regex_path.replace("\\(assay\\)", "(?P<assay>.*?)")
    regex_path = regex_path.replace("\\(channel\\)", "(?P<channel>.*?)")
    regex_path = re.sub(r"\\\(time\s*\|?.*?\\\)", r"(?P<time>.*?)", regex_path)
    regex_path = regex_path.replace("\\(row\\)", "(?P<row>.*?)")
    regex_path = regex_path.replace("\\(col\\)", "(?P<col>.*?)")
    regex_path = re.compile(regex_path, re.IGNORECASE)

    path_dict = collections.defaultdict(dict)
    for path in paths:
        match = regex_path.fullmatch(path)
        if "(assay)" in pattern:
            assay = match.group("assay")
        else:
            assay = ""

        if "(channel)" in pattern:
            channel = match.group("channel")
        else:
            channel = -1

        time_search = re.search(r"\(time\s*\|?\s*(.*?)\)", pattern)
        if time_search:
            format_str = time_search.group(1)
            if not format_str:
                format_str = "%Y%m%d-%H%M%S"
            time_str = match.group("time")
            time = datetime.datetime.strptime(time_str, format_str)
        else:
            time = -1

        if "(row)" in pattern:
            row = int(match.group("row"))
        else:
            row = -1

        if "(col)" in pattern:
            col = int(match.group("col"))
        else:
            col = -1

        idx = (channel, time, row, col)
        if idx not in path_dict[assay]:
            path_dict[assay][idx] = path
        else:
            raise ValueError(f"{path} and {path_dict[assay][idx]} map to the same index.")

    return path_dict
