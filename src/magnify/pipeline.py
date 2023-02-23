from typing import Callable

from numpy.typing import ArrayLike

from magnify.assay import Assay


class Pipeline:
    def __init__(self):
        self.loader: Callable[[ArrayLike | str], Assay] = []
        self.components: Callable[[Assay], Assay] = []

    def __call__(self, data: ArrayLike | str) -> Assay:
        assays = []
        for assay in self.loader(data):
            for name, component in self.components:
                assay = component(assay)
            assays.append(assay)

        return Assay.concat_time(assays)

    def add_pipe(self, name: str) -> None:
        self.components.append((name, Pipeline.get_component(name)))

    registered_components: dict[str, Callable[[Assay], Assay]] = {}
    registered_factories: dict[str, Callable[[], Callable[[Assay], Assay]]] = {}

    @classmethod
    def get_component(cls, name):
        if name in cls.registered_components:
            return cls.registered_components[name]
        else:
            return cls.registered_factories[name]()

    @classmethod
    def factory(cls, name):
        def decorator(func: Callable[[Assay], Assay]):
            cls.registered_factories[name] = func
            return func

        return decorator

    @classmethod
    def component(cls, name: str):
        def decorator(func: Callable[[Assay], Assay]):
            cls.registered_components[name] = func
            return func

        return decorator
