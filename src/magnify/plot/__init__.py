__all__ = [
    "imshow",
    "mrbles_clusters",
    "ndplot",
    "relplot",
    "roishow",
    "set_style",
    "display_edge_detection",
]
from magnify.plot.image import imshow, roishow
from magnify.plot.mrbles import mrbles_clusters
from magnify.plot.ndplot import ndplot
from magnify.plot.relation import relplot
from magnify.plot.style import set_style
from magnify.plot.vis import display_edge_detection

set_style()
