from __future__ import annotations

import fnmatch
import glob
import os
import pathlib
import re
from collections.abc import Iterator, Sequence

import xarray as xr

import magnify.registry as registry
import magnify.utils as utils


class Reader:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        data: str | xr.DataArray | xr.Dataset | Sequence[str | xr.DataArray | xr.Dataset],
    ) -> Iterator[xr.Dataset]:
        data = [data] if isinstance(data, utils.PathLike | xr.DataArray | xr.Dataset) else data
        for d in data:
            if isinstance(d, xr.Dataset | xr.DataArray):
                yield d
                continue

            path_dict = extract_paths(d)
            if len(path_dict) == 0:
                raise FileNotFoundError(f"The pattern {d} did not lead to any files.")

            for name in sorted(path_dict, key=lambda n: utils.natural_sort_key(n or "")):
                path = pathlib.Path(path_dict[name])
                # The last element in the path is the zarr group directory.
                try:
                    xp = xr.open_zarr(path.parent, group=path.name, consolidated=False)
                except Exception as e:
                    raise ValueError(
                        f"Could not read {path} as a zarr group (matched by pattern {d!r}). "
                        "Each match should be a zarr group directory written by magnify."
                    ) from e
                xp.attrs["name"] = name or ""

                yield xp

    @registry.readers.register("read")
    def make():
        return Reader()


def extract_paths(pattern: str) -> dict[str | None, str]:
    """Expand a glob pattern and extract experiment names.

    The pattern may contain a single named group ``(assay)`` marking the part of the path that names
    the experiment. Returns a mapping from experiment name (``None`` when no name group is given) to
    the matched absolute path.
    """
    pattern = os.path.expanduser(pattern)
    # Turn the (assay) group into a glob wildcard and a named regex capture group.
    glob_path = re.sub(r"\(assay.*?\)", "*", pattern)
    regex_path = re.compile(
        re.sub(r"\\\(assay.*?\\\)", r"(?P<assay>[^/\\]*?)", fnmatch.translate(pattern)),
        re.IGNORECASE,
    )

    path_dict: dict[str | None, str] = {}
    for path in glob.glob(glob_path, recursive=True):
        match = regex_path.fullmatch(path)
        name = match.group("assay") if match and "assay" in regex_path.groupindex else None
        abspath = os.path.abspath(path)
        if name in path_dict:
            raise ValueError(f"{abspath} and {path_dict[name]} map to the same name {name!r}.")
        path_dict[name] = abspath
    return path_dict
