from __future__ import annotations
from typing import Callable

from numpy.typing import ArrayLike
import tqdm

from magnify.assay import Assay
import magnify.registry as registry
import magnify.utils as utils


class Pipeline:
    def __init__(self, reader: str):
        self.reader: Callable[[ArrayLike | str], Assay] = registry.readers.get(reader)()
        self.components: list[Callable[[Assay], Assay]] = []

    def __call__(
        self,
        data: ArrayLike | str,
        names: ArrayLike | str = None,
        search_on: str = "egfp",
        times: Sequence[int] | None = None,
        channels: Sequence[str] | None = None,
        progress_bar: bool = False,
        **kwargs,
    ) -> Assay:
        inputs = self.reader(
            data=data, names=names, search_on=search_on, times=times, channels=channels
        )
        assays = []
        for assay in tqdm.tqdm(inputs, disable=not progress_bar):
            for name, component in self.components:
                assay = component(assay, **utils.valid_kwargs(kwargs, component))

            assays.append(assay)

        if len(assays) == 1:
            assays = assays[0]

        return assays

    def add_pipe(self, name: str) -> None:
        self.components.append((name, registry.components.get(name)()))
