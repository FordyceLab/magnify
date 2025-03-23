from typing import Callable, Sequence

import xarray as xr
from numpy.typing import ArrayLike

import magnify.registry as registry


class Pipeline:
    def __init__(self, reader: str):
        self.reader: Callable[[ArrayLike | str], xr.Dataset] = registry.readers.get(reader)()
        self.components: list[tuple[str, Callable[[xr.Dataset], xr.Dataset]]] = []

    def __call__(
        self,
        data: ArrayLike | str,
    ) -> xr.Dataset | list[xr.Dataset]:
        inputs = self.reader(data=data)
        assays = []
        for assay in inputs:
            for _, component in self.components:
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
        **kwargs,
    ) -> None:
        component_factory = registry.components.get(name)
        component = component_factory(**kwargs)

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
        else:
            raise ValueError("before/after must be a string or int.")

        self.components.insert(idx, new_element)

    def remove_pipe(self, name: str) -> None:
        idx = list(zip(*self.components))[0].index(name)
        self.components.pop(idx)
