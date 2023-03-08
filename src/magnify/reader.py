from __future__ import annotations
import datetime
import fnmatch
import glob
import os
import re

from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
import tifffile

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
            data = os.path.expanduser(data)

            regex_path = fnmatch.translate(data)
            regex_path = regex_path.replace("\\(time\\)", "(?P<time>.*?)")
            regex_path = regex_path.replace("\\(channel\\)", "(?P<channel>.*?)")
            regex_path = regex_path.replace("\\(row\\)", "(?P<row>.*?)")
            regex_path = regex_path.replace("\\(col\\)", "(?P<col>.*?)")
            regex_path = re.compile(regex_path, re.IGNORECASE)

            glob_path = data
            glob_path = glob_path.replace("(time)", "*")
            glob_path = glob_path.replace("(channel)", "*")
            glob_path = glob_path.replace("(row)", "*")
            glob_path = glob_path.replace("(col)", "*")

            # Search for files matching the pattern.
            paths = glob.glob(glob_path, recursive=True)
            if len(paths) == 0:
                raise FileNotFoundError(f"The pattern {data} did not lead to any files.")

            path_dict = {}
            for path in paths:
                match = regex_path.fullmatch(path)
                if "(time)" in data:
                    time_str = match.group("time")
                    time = datetime.datetime.strptime(time_str, "%Y%m%d-%H%M%S").timestamp()
                    if times is not None and time not in times:
                        continue
                else:
                    time = 0

                if "(channel)" in data:
                    channel = match.group("channel")
                else:
                    channel = "?"

                if "(row)" in data:
                    row = int(match.group("row"))
                else:
                    row = 0

                if "(col)" in data:
                    col = int(match.group("col"))
                else:
                    col = 0

                idx = (time, channel, row, col)
                if idx not in path_dict:
                    path_dict[idx] = path
                else:
                    raise ValueError(f"{path} and {path_dict[idx]} map to the same index.")

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
    @staticmethod
    def make():
        return ChipReader()


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
