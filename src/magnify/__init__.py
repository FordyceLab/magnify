__version__ = "0.2.0"

__all__ = [
    "load",
]
from magnify.registry import load
import magnify.find
import magnify.postprocess
import magnify.preprocess
import magnify.reader
import magnify.stitch
