from __future__ import annotations
from typing import Callable

from numpy.typing import ArrayLike
import catalogue

from magnify.pipeline import Pipeline

readers = catalogue.create("magnify", "readers")
components = catalogue.create("magnify", "components")


def load(name: str) -> Pipeline:
    if name == "chip":
        return chip_pipeline()
    elif name == "mrbles":
        return mrbles_pipeline()
    elif name == "beads":
        return mrbles_pipeline()
    elif name == "imread":
        return imread_pipeline()
    else:
        raise ValueError(f"Pipeline {name} does not exist.")


def chip_pipeline():
    pipe = Pipeline("read")
    pipe.add_pipe("read_pinlist")
    # pipe.add_pipe("preprocessor")
    pipe.add_pipe("flip_horizontal")
    pipe.add_pipe("stitch")
    pipe.add_pipe("find_buttons")
    # pipe.add_pipe("background_filter")

    return pipe


def mrbles_pipeline():
    pipe = Pipeline("read")
    pipe.add_pipe("flatfield_correct")
    pipe.add_pipe("stitch")
    pipe.add_pipe("find_beads")
    # pipe.add_pipe("background_filter")

    return pipe


def imread_pipeline():
    pipe = Pipeline("read")
    # pipe.add_pipe("flip_horizontal")
    # pipe.add_pipe("stitch")
    return pipe
