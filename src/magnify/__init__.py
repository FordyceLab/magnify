__version__ = "0.1.6"

__all__ = [
    "Assay",
    "load",
]
from magnify.assay import Assay
from magnify.registry import load
import magnify.preprocess
import magnify.reader
