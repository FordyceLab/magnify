from __future__ import annotations

import logging
from typing import Callable, Sequence

import confection
import xarray as xr
from numpy.typing import ArrayLike

import magnify.logger as logger
import magnify.registry as registry
import magnify.utils as utils


class Pipeline:
    def __init__(self, reader: str, config: dict[str, dict[str, str]]):
        self.reader: Callable[[ArrayLike | str], xr.Dataset] = registry.readers.get(reader)()
        self.config = confection.Config(config)
        self.components: list[Callable[[xr.Dataset], xr.Dataset]] = []

        if "debug" in self.config and self.config["debug"]:
            logger.log_level = logging.DEBUG
        else:
            logger.log_level = logging.INFO

    def __call__(
        self,
        data: ArrayLike | str,
        times: Sequence[int] | None = None,
        channels: Sequence[str] | None = None,
    ) -> xr.Dataset | list[xr.Dataset]:
        inputs = self.reader(data=data, times=times, channels=channels)
        assays = []
        for assay in inputs:
            for name, component in self.components:
                assay = component(assay)

            assays.append(assay)

        if len(assays) == 1:
            assays = assays[0]

        return assays

    def add_pipe(
        self,
        name: str,
        after: str | int | None = None,
        before: str | int | None = None,
        first: bool = False,
        last: bool = False,
    ) -> None:
        component_factory = registry.components.get(name)
        component = component_factory(**utils.valid_kwargs(self.config, component_factory))

        if after is None and before is None and not first and not last:
            last = True
        if (after is not None) + (before is not None) + first + last > 1:
            raise ValueError("Only one of after, before, first, and last can be set.")
        new_element = (name, component)
        if first:
            idx = 0
        elif last:
            idx = len(self.components)
        elif isinstance(before, int):
            idx = before
        elif isinstance(before, str):
            idx = [name for (name, _) in self.components].index(before)
        elif isinstance(after, int):
            idx = after + 1
        elif isinstance(after, str):
            idx = [name for (name, _) in self.components].index(after) + 1

        self.components.insert(idx, new_element)

    def remove_pipe(self, name: str) -> None:
        idx = list(zip(*self.components))[0].index(name)
        self.components.pop(idx)
