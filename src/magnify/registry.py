from __future__ import annotations
import functools
import inspect
from typing import Callable

from numpy.typing import ArrayLike
import catalogue
import confection

from magnify.pipeline import Pipeline

readers = catalogue.create("magnify", "readers")
components = catalogue.create("magnify", "components")


def load(name: str, **kwargs) -> Pipeline:
    if name == "ps-chip":
        return ps_chip_pipeline(**kwargs)
    elif name == "pc-chip":
        return pc_chip_pipeline(**kwargs)
    elif name == "mrbles":
        return mrbles_pipeline(**kwargs)
    elif name == "beads":
        return mrbles_pipeline(**kwargs)
    elif name == "imread":
        return imread_pipeline(**kwargs)
    else:
        raise ValueError(f"Pipeline {name} does not exist.")


def component(name):
    def component_decorator(func):
        @functools.wraps(func)
        def component_factory(*args, **kwargs):
            return functools.partial(func, *args, **kwargs)

        # Make the factory signature identical to func's except for the first argument.
        signature = inspect.signature(func)
        signature = signature.replace(parameters=list(signature.parameters.values())[1:])
        component_factory.__signature__ = signature
        components.register(name)(component_factory)
        return func

    return component_decorator


def ps_chip_pipeline(**kwargs):
    # Button centers are apart 375um vertically and 655um horizontally.
    # Assuming a 4x objective and 2x2 binning each pixel is 3.22um.
    defaults = confection.Config(dict(row_dist=375 / 3.22, col_dist=655 / 3.22))
    config = defaults.merge(confection.Config(kwargs))
    pipe = Pipeline("read", config=config)
    pipe.add_pipe("read_pinlist")
    # pipe.add_pipe("preprocessor")
    pipe.add_pipe("horizontal_flip")
    pipe.add_pipe("stitch")
    pipe.add_pipe("horizontal_flip")
    pipe.add_pipe("vertical_flip")
    pipe.add_pipe("find_buttons")
    # pipe.add_pipe("background_filter")
    pipe.add_pipe("squeeze")
    pipe.add_pipe("summarize_sum")

    return pipe


def pc_chip_pipeline(**kwargs):
    # Button centers are apart 412um vertically and 760um horizontally.
    # Assuming a 4x objective and 2x2 binning each pixel is 3.22um.
    defaults = confection.Config(dict(row_dist=412 / 3.22, col_dist=760 / 3.22))
    config = defaults.merge(confection.Config(kwargs))
    pipe = Pipeline("read", config=config)
    pipe.add_pipe("read_pinlist")
    # pipe.add_pipe("preprocessor")
    pipe.add_pipe("horizontal_flip")
    pipe.add_pipe("stitch")
    pipe.add_pipe("horizontal_flip")
    pipe.add_pipe("vertical_flip")
    pipe.add_pipe("find_buttons")
    # pipe.add_pipe("background_filter")
    pipe.add_pipe("squeeze")
    pipe.add_pipe("summarize_sum")

    return pipe


def mrbles_pipeline(**kwargs):
    pipe = Pipeline("read", config=kwargs)
    pipe.add_pipe("flatfield_correct")
    pipe.add_pipe("stitch")
    pipe.add_pipe("find_beads")
    # pipe.add_pipe("background_filter")
    pipe.add_pipe("squeeze")

    return pipe


def imread_pipeline(**kwargs):
    pipe = Pipeline("read", config=kwargs)
    pipe.add_pipe("horizontal_flip")
    pipe.add_pipe("stitch")
    pipe.add_pipe("squeeze")
    return pipe
