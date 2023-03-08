from __future__ import annotations
from typing import Callable

from numpy.typing import ArrayLike
import catalogue

from magnify.pipeline import Pipeline

readers = catalogue.create("magnify", "readers")
components = catalogue.create("magnify", "components")


def load(name: str):
    if name == "chip":
        return chip_pipeline()
    else:
        raise ValueError(f"Pipeline {name} does not exist.")


def chip_pipeline():
    pipe = Pipeline("chip_reader")
    pipe.add_pipe("preprocessor")
    pipe.add_pipe("horizontal_flip")
    pipe.add_pipe("stitcher")
    pipe.add_pipe("button_finder")
    pipe.add_pipe("button_segmenter")
    # pipe.add_pipe("background_filter")

    return pipe
