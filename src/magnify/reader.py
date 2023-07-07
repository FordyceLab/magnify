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
import bs4
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
        times: Sequence[int] | None = None,
        channels: Sequence[str] | None = None,
    ) -> Iterator[xr.Dataset]:
        if isinstance(data, str):
            data = [data]

        for d in data:
            path_dict, meta_dict = extract_paths(
                d, assay="str", channel="str", time="time", row="int", col="int"
            )
            if len(path_dict) == 0:
                raise FileNotFoundError(f"The pattern {d} did not lead to any files.")

            # None assay names should just mean a nameless assay.
            path_dict = {("",) + k[1:] if k[0] is None else k: v for k, v in path_dict.items()}
            assay_names = set(list(zip(*path_dict.keys()))[0])

            for assay_name in sorted(assay_names, key=utils.natural_sort_key):
                # Extract the paths for this assay and turn all Nones into -1 so sorting works.
                assay_dict = {
                    tuple(-1 if x is None else x for x in k[1:]): v
                    for k, v in path_dict.items()
                    if k[0] == assay_name
                }
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
                    dtype = tif.series[0].dtype
                    inner_shape = tif.series[0].shape
                    page_shape = tif.pages[0].shape

                    letter_to_dim = {
                        "C": "channel",
                        "T": "time",
                        "Z": "depth",
                        "Y": "tile_y",
                        "X": "tile_x",
                        "R": "tile_pos",
                    }
                    dims_in_file = [letter_to_dim[c] for c in tif.series[0].axes]

                    if (
                        time_coords is None
                        and tif.is_micromanager
                        and "StartTime" in tif.micromanager_metadata["Summary"]
                    ):
                        # Get the time string without timezone info.
                        time_str = tif.micromanager_metadata["Summary"]["StartTime"][:-6]
                        start_time = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
                        if "time" in dims_in_file:
                            # Only look at the first file's description since time isn't across multiple files.
                            planes = [
                                pl
                                for pl in bs4.BeautifulSoup(tif.pages[0].description, "xml")
                                .find("Image")
                                .find_all("Plane")
                            ]
                            assert all(pl.get("DeltaTUnit") == "ms" for pl in planes)
                            time_coords = [
                                start_time
                                + datetime.timedelta(milliseconds=float(pl.get("DeltaT")))
                                for pl in planes
                            ]
                            if "channel" in dims_in_file:
                                stride = inner_shape[dims_in_file.index("channel")]
                            else:
                                stride = 1
                            assert len(time_coords) % stride == 0
                            time_coords = time_coords[::stride]
                        else:
                            time_coords = [start_time]
                    if channel_coords is None and tif.is_micromanager:
                        channel_coords = tif.micromanager_metadata["Summary"]["ChNames"]

                    if "tile_pos" in dims_in_file:
                        # Tiles are always saved in multiple files so ignore this dimension
                        # since the user should specify tiles in their search path.
                        tile_idx = dims_in_file.index("tile_pos")
                        inner_shape = inner_shape[:tile_idx] + inner_shape[tile_idx + 1 :]
                        dims_in_file = dims_in_file[:tile_idx] + dims_in_file[tile_idx + 1 :]

                    if "depth" in dims_in_file:
                        raise ValueError("tiff files with a Z dimension are not yet supported.")
                    if "tile_y" not in dims_in_file or "tile_x" not in dims_in_file:
                        raise ValueError("tiff files must contain an X and Y dimension.")

                # Check the dimensions specified in the path and inside the tiff file do not overlap.
                if set(dims_in_file).intersection(dims_in_path):
                    raise ValueError(
                        "Dimensions specified in the path names and inside the tiff file overlap."
                    )

                # Setup a dask array to lazyload image tiles and extract attributes.
                filenames = [x for _, x in sorted(assay_dict.items())]

                def read_tile(block_id, filenames):
                    outer_id = block_id[: len(outer_shape)]
                    inner_id = block_id[len(outer_shape) :]
                    if len(outer_id) > 0:
                        file_idx = np.ravel_multi_index(outer_id, outer_shape)
                    else:
                        # This is the case where we don't have indices outside the file.
                        file_idx = 0

                    with tifffile.TiffFile(filenames[file_idx]) as tif:
                        page_ndim = len(page_shape)
                        if len(inner_shape) > page_ndim:
                            page_idx = np.ravel_multi_index(
                                inner_id[:-page_ndim], inner_shape[:-page_ndim]
                            )
                        else:
                            # This is the case where no dimensions correspond to page dims.
                            page_idx = 0
                        # Read the page from disk.
                        page = tif.pages[page_idx].asarray()
                        # Expand the dimensions of the block to incorporate outer dims and page index dims.
                        return np.expand_dims(page, axis=tuple(range(len(block_id) - page_ndim)))

                # Chunk the images so that each chunk represents a single page in the tiff file.
                tiles = da.map_blocks(
                    functools.partial(read_tile, filenames=filenames),
                    dtype=dtype,
                    chunks=(
                        (
                            tuple((1,) * size for size in outer_shape)
                            + tuple((1,) * size for size in inner_shape[: -len(page_shape)])
                            + inner_shape[-len(page_shape) :]
                        )
                    ),
                )

                coords = {}
                # Set named coordinates for channels and times if they're available.
                if channel_coords is not None:
                    coords["channel"] = channel_coords
                if time_coords is not None:
                    coords["time"] = time_coords

                # if "time" in coords:
                #     coords["time"] = pd.to_datetime(coords["time"])
                #     coords["time"] -= coords["time"][0]

                # Put all our data into an xarray dataset.
                assay = xr.Dataset(
                    {"tile": (dims_in_path + dims_in_file, tiles)},
                    coords=coords,
                    attrs={"name": assay_name},
                )

                # Make sure the assay always has a time and channel dimension.
                if "channel" not in assay.dims:
                    assay = assay.expand_dims("channel", 0)
                if "time" not in assay.dims:
                    assay = assay.expand_dims("time", 1)

                # Reorder the dimensions so they're always consistent and add missing dimensions.
                desired_order = ["channel", "time", "tile_row", "tile_col", "tile_y", "tile_x"]
                for dim in desired_order:
                    if dim not in assay.tile.dims:
                        assay["tile"] = assay.tile.expand_dims(dim)

                assay = assay.transpose(*desired_order)

                # Add any metadata coodinates specified by the user.
                for (meta_name, dim), meta_idxs_dict in meta_dict.items():
                    if dim == "time":
                        # Make sure we're indexing using datetimes.
                        dim_idxs = assay[dim] = pd.to_datetime(assay[dim])
                    else:
                        dim_idxs = assay[dim].values

                    meta_idxs = [meta_idxs_dict[dim_idx] for dim_idx in dim_idxs]
                    assay = assay.assign_coords({meta_name: (dim, meta_idxs)})

                # Make sure the time dimension starts at 0 seconds.
                assay = assay.assign_coords(time=assay.time - assay.time[0])

                yield assay

    @registry.readers.register("read")
    def make():
        return Reader()


def extract_paths(pattern, **kwargs) -> dict[tuple[int, str, int, int], str]:
    default_formatters = {
        "": lambda x, y: x,
        "str": lambda x, y: x,
        "time": lambda x, y: datetime.datetime.strptime(x, y if y else "%Y%m%d-%H%M%S"),
        "int": lambda x, y: int(x),
        "float": lambda x, y: float(x),
    }

    keys = kwargs
    # Make sure keys is always a dict whose values are formatting functions.
    if not isinstance(keys, dict):
        keys = {key: "str" for key in keys}
    keys = {k: f if callable(f) else default_formatters[f] for k, f in keys.items()}
    all_keys = list(keys)

    # Format the pattern so we can do a glob search and so we can use it to extract
    # filename metadata using regexes.
    pattern = os.path.expanduser(pattern)
    meta = collections.defaultdict(dict)
    glob_path = pattern
    regex_path = fnmatch.translate(pattern)
    for key, formatter in list(keys.items()):
        # For globs replace all named search patterns e.g.: (time|%S) with the wildcard *.
        glob_path = re.sub(rf"\({key}.*?\)", "*", glob_path)
        glob_path = re.sub(rf"\(.*?_{key}.*?\)", "*", glob_path)
        # For regexes replace all named search patterns with named wildcard groups.
        regex_path = re.sub(rf"\\\({key}.*?\\\)", rf"(?P<{key}>.*?)", regex_path)
        regex_path = re.sub(rf"\\\((.*?)_{key}.*?\\\)", r"(?P<\1>.*?)", regex_path)

        # Get any associated formatting information in the named search pattern.
        key_search = re.search(rf"\({key}(?:\s*\|\s*(.*?))?\)", pattern)
        if key_search:
            format_str = key_search.group(1)
            # Rebind the function, we need to set default argument values because of closure rules.
            keys[key] = lambda x, y=format_str, f=formatter: f(x, y)
        else:
            # Remove keys that weren't specified in the pattern.
            del keys[key]

        # Get meta information that provides alternate value mappings for a given key
        # along with any formatting information e.g. (concentration_time|float).
        meta_search = re.findall(rf"\((.*?)_{key}(?:\s*\|\s*(.*?))?(?:\s*\|\s*(.*?))?\)", pattern)
        for name, formatter_str, format_str in meta_search:
            meta_formatter = default_formatters[formatter_str]
            # Once again we need to set default arguments because of closure rules in Python.
            meta_formatter = lambda x, y=format_str, f=meta_formatter: f(x, y)
            meta[key][name] = meta_formatter

    regex_path = re.compile(regex_path, re.IGNORECASE)
    # Search for files matching the pattern.
    paths = glob.glob(glob_path, recursive=True)

    path_dict = {}
    meta_dict = collections.defaultdict(dict)
    for path in paths:
        match = regex_path.fullmatch(path)
        idxs = []
        for key in all_keys:
            if key in keys:
                # First get the value for the given key for the current path.
                idx = match.group(key)
                formatter = keys[key]
                idx = formatter(idx)
                idxs.append(idx)
                # Then get all the associated metadata for that key.
                for name, formatter in meta[key].items():
                    meta_idx = match.group(name)
                    meta_dict[name, key][idx] = formatter(meta_idx)
            else:
                # If the key wasn't specified in the pattern we still include it in the idxs.
                idxs.append(None)

        # Keep track of which indices correspond to which paths.
        idxs = tuple(idxs)
        if idxs not in path_dict:
            path_dict[idxs] = os.path.abspath(path)
        else:
            raise ValueError(f"{path} and {path_dict[idxs]} map to the same index.")

    return path_dict, meta_dict
