from __future__ import annotations
from collections.abc import Callable, Iterable
from typing import Any
import inspect
import re

import cv2 as cv
import numba
import numpy as np
import xarray as xr


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(float)
    arr = arr - np.min(arr)
    if np.max(arr) > 0:
        arr = 255 * arr / np.max(arr)
    return arr.astype(np.uint8)


def circle(image_length: int, row: int, col: int, radius: int, value: Any = 1.0) -> np.ndarray:
    image = np.zeros((image_length, image_length), dtype=np.uint8)
    cv.circle(image, (col, row), radius, 1, -1)
    image = image.astype(type(value)) * value
    return image


def annulus(
    image_length: int, row: int, col: int, outer_radius: int, inner_radius: int, value: Any = 1.0
) -> np.ndarray:
    outer_circle = circle(image_length, row, col, outer_radius, value)
    inner_circle = circle(image_length, row, col, inner_radius, value)
    return outer_circle & ~inner_circle


def contour_center(contour):
    m = cv.moments(contour)
    return m["m01"] / m["m00"], m["m10"] / m["m00"]


@numba.jit(nopython=True)
def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


@numba.jit(nopython=True)
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


def natural_sort_key(s: str) -> list[str]:
    reg = re.compile("([0-9]+)")
    return [int(text) if text.isdigit() else text.lower() for text in reg.split(s)]


def to_list(x: Any) -> list:
    if x is None:
        return []
    elif not isinstance(x, Iterable) or isinstance(x, str):
        return [x]
    else:
        return list(x)
