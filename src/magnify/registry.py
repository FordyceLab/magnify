from __future__ import annotations
import functools
import inspect
from typing import Callable

from numpy.typing import ArrayLike
import catalogue
import confection
from functools import wraps

from magnify.pipeline import Pipeline

readers = catalogue.create("magnify", "readers")
components = catalogue.create("magnify", "components")


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

def pipeline_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract parameters from args or kwargs (depending on how they are passed)
        data = kwargs.get('data', None) if 'data' in kwargs else None
        times = kwargs.get('data', None) if 'times' in kwargs else None
        channels = kwargs.get('data', None) if 'channels' in kwargs else None
        return_pipe = kwargs.get('return_pipe', False) if 'return_pipe' in kwargs else False

        # Input check: If pipe is to be executed, a path to data must be provided.
        if not return_pipe and data is None:
            raise ValueError("The 'data' parameter cannot be None when 'return_pipe' is False.")

        pipe = func(*args, **kwargs)

        if return_pipe:
            return pipe 
        else:
            return pipe(data=data, times=times, channels=channels)
    return wrapper

@pipeline_wrapper
def mini_chip(**kwargs):
    # Button centers are apart 375um vertically and 655um horizontally.
    # Assuming a 4x objective and 1x1 binning each pixel is 1.61um.
    defaults = confection.Config(dict(row_dist=375 / 1.61, col_dist=400 / 1.61, shape=(8, 8)))
    config = defaults.merge(confection.Config(kwargs))
    pipe = Pipeline("read", config=config)
    pipe.add_pipe("identify_buttons")
    pipe.add_pipe("stitch")
    pipe.add_pipe("rotate")
    pipe.add_pipe("find_buttons")
    pipe.add_pipe("filter_expression")
    pipe.add_pipe("filter_nonround")
    pipe.add_pipe("filter_leaky")
    pipe.add_pipe("drop")

    return pipe

@pipeline_wrapper
def ps_chip(**kwargs):
    # Button centers are apart 375um vertically and 655um horizontally.
    # Assuming a 4x objective and 2x2 binning each pixel is 3.22um.
    defaults = confection.Config(dict(row_dist=375 / 3.22, col_dist=655 / 3.22))
    config = defaults.merge(confection.Config(kwargs))
    pipe = Pipeline("read", config=config)
    pipe.add_pipe("identify_buttons")
    pipe.add_pipe("stitch")
    pipe.add_pipe("rotate")
    pipe.add_pipe("find_buttons")
    pipe.add_pipe("filter_expression")
    pipe.add_pipe("filter_nonround")
    pipe.add_pipe("filter_leaky")
    pipe.add_pipe("drop")

    return pipe

@pipeline_wrapper
def pc_chip(**kwargs):
    # Button centers are apart 412um vertically and 760um horizontally.
    # Assuming a 4x objective and 2x2 binning each pixel is 3.22um.
    defaults = confection.Config(dict(row_dist=406 / 3.22, col_dist=750 / 3.22))
    config = defaults.merge(confection.Config(kwargs))
    pipe = Pipeline("read", config=config)
    pipe.add_pipe("identify_buttons")
    pipe.add_pipe("horizontal_flip")
    pipe.add_pipe("stitch")
    pipe.add_pipe("horizontal_flip")
    pipe.add_pipe("rotate")
    pipe.add_pipe("find_buttons")
    pipe.add_pipe("filter_expression")
    pipe.add_pipe("filter_nonround")
    pipe.add_pipe("filter_leaky")
    pipe.add_pipe("drop")

    return pipe

@pipeline_wrapper
def mrbles(**kwargs):
    pipe = Pipeline("read", config=kwargs)
    pipe.add_pipe("flatfield_correct")
    pipe.add_pipe("stitch")
    pipe.add_pipe("find_beads")
    pipe.add_pipe("identify_mrbles")
    pipe.add_pipe("drop")

    return pipe

@pipeline_wrapper
def beads(**kwargs):
    pipe = Pipeline("read", config=kwargs)
    pipe.add_pipe("flatfield_correct")
    pipe.add_pipe("stitch")
    pipe.add_pipe("find_beads")
    pipe.add_pipe("drop")

    return pipe

@pipeline_wrapper
def image(**kwargs):
    pipe = Pipeline("read", config=kwargs)
    pipe.add_pipe("stitch")
    pipe.add_pipe("rotate")
    pipe.add_pipe("drop")
    return pipe
