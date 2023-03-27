from __future__ import annotations
from collections.abc import Callable
from typing import Any
import inspect

import cv2 as cv
import numpy as np


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(float)
    arr = 255 * arr / np.max(arr)
    return arr.astype(np.uint8)


def circle(image_length: int, row: int, col: int, radius: int, value: Any = 1.0) -> np.ndarray:
    image = np.zeros((image_length, image_length), dtype=np.uint8)
    cv.circle(image, (col, row), radius, 1, -1)
    image = image.astype(type(value)) * value
    return image


def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


def bounding_box(
    x: int, y: int, box_length: int, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    top = y - box_length // 2
    bottom = y + ceildiv(box_length, 2)
    if top < 0:
        bottom += -top
        top = 0
    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height
    left = x - box_length // 2
    right = x + ceildiv(box_length, 2)
    if left < 0:
        right += -left
        left = 0
    if right > image_width:
        left -= right - image_width
        right = image_width
    return top, bottom, left, right


def valid_kwargs(kwargs: dict[str, Any], func: Callable) -> dict[str, Any]:
    args = list(inspect.signature(func).parameters)
    return {k: kwargs[k] for k in kwargs if k in args}
