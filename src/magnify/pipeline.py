from __future__ import annotations
from typing import Callable

import confection
from numpy.typing import ArrayLike
import xarray as xr

import magnify.registry as registry
import magnify.utils as utils


class Pipeline:
    def __init__(self, reader: str, config: dict[str, dict[str, str]]):
        self.reader: Callable[[ArrayLike | str], xr.Dataset] = registry.readers.get(reader)()
        self.config = confection.Config(config)
        self.components: list[Callable[[xr.Dataset], xr.Dataset]] = []

    def __call__(
        self,
        data: ArrayLike | str,
        names: ArrayLike | str = None,
        search_on: str = "egfp",
        times: Sequence[int] | None = None,
        channels: Sequence[str] | None = None,
    ) -> xr.Dataset | list[xr.Dataset]:
        inputs = self.reader(
            data=data, names=names, search_on=search_on, times=times, channels=channels
        )
        assays = []
        for assay in inputs:
            for name, component in self.components:
                assay = component(assay)

            assays.append(assay)

        if len(assays) == 1:
            assays = assays[0]

        return assays

    def add_pipe(self, name: str) -> None:
        component_factory = registry.components.get(name)
        component = component_factory(**utils.valid_kwargs(self.config, component_factory))
        self.components.append((name, component))
