from numpy.typing import ArrayLike

from magnify.assay import Assay
from magnify.pipeline import Pipeline


def loader(data: ArrayLike | str) -> Assay:
    if isinstance(data, str):

    else:
        return Assay(data)

    return Assay()
