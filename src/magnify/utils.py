from collections.abc import Callable
from numbers import Number
from typing import Any
import inspect

import cv2 as cv
import numpy as np


def to_uint8(arr: np.ndarray):
    arr = arr.astype(float)
    arr = 255 * arr / np.max(arr)
    return arr.astype(np.uint8)


def circle(image_length: int, row: int, col: int, radius: int, value: Number = 1.0):
    image = np.zeros((image_length, image_length), dtype=np.uint8)
    cv.circle(image, (col, row), radius, 1, -1)
    image = image.astype(type(value)) * value
    return image


def ceildiv(a: int, b: int):
    return -(a // -b)


def bounding_box(row: int, col: int, box_length: int):
    top = row - box_length // 2
    bottom = row + ceildiv(box_length, 2)
    left = col - box_length // 2
    right = col + ceildiv(box_length, 2)
    return top, bottom, left, right


def valid_kwargs(kwargs: dict[str, Any], func: Callable):
    args = list(inspect.signature(func).parameters)
    return {k: kwargs[k] for k in kwargs if k in args}
