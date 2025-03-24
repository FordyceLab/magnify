from typing import Callable

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
        component: str | Callable[..., xr.Dataset],
        name: str | None = None,
        after: str | int | None = None,
        before: str | int | None = None,
        first: bool = False,
        last: bool = False,
        **kwargs,
    ) -> None:
        if isinstance(component, str):
            if name is None:
                name = component
            component_factory = registry.components.get(component)
            func = component_factory(**kwargs)
        else:
            name = component.__name__ if name is None else name

            def func(xp):
                return component(xp, **kwargs)

        if after is None and before is None and not first and not last:
            last = True
        if (after is not None) + (before is not None) + first + last > 1:
            raise ValueError("Only one of after, before, first, and last can be set.")

        # Check that the new component's name is unique
        if self.components and name in next(zip(*self.components)):
            raise ValueError(f"A component with the name '{name}' already exists in the pipeline.")

        # Find where to insert the component.
        new_element = (name, func)
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
