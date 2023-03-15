from __future__ import annotations
from typing import Callable

from numpy.typing import ArrayLike
import tqdm

from magnify.assay import Assay
import magnify.registry as registry


class Pipeline:
    def __init__(self, reader: str):
        self.reader: Callable[[ArrayLike | str], Assay] = registry.readers.get(reader)()
        self.components: list[Callable[[Assay], Assay]] = []

    def __call__(
        self,
        data: ArrayLike | str,
        names: ArrayLike | str,
        search_on: str = "egfp",
        times: Sequence[int] | None = None,
        channels: Sequence[str] | None = None,
        progress_bar: bool = False,
    ) -> Assay:
        assays = []
        for assay in tqdm.tqdm(
            self.reader(data=data, names=names, search_on=search_on, times=times, channels=channels)
        ):
            for name, component in self.components:
                assay = component(assay)
            assays.append(assay)

        return Assay.from_assays(assays)

    def add_pipe(self, name: str) -> None:
        self.components.append((name, registry.components.get(name)()))
