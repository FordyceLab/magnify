__version__ = "0.1.6"

__all__ = [
    "Assay",
    "find_buttons",
    "segment_buttons",
    "overlap_stitch",
    "pipe",
    "basic_correct_tiles",
]
from magnify.assay import Assay
from magnify.find import find_buttons
from magnify.segment import segment_buttons
from magnify.stitch import overlap_stitch
from magnify.pipeline import pipe
from magnify.preprocess import basic_correct_tiles
