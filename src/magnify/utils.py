from __future__ import annotations

import inspect
import os
import re
from collections.abc import Callable, Iterable
from typing import Any

import cv2 as cv
import numba
import numpy as np

PathLike = str | bytes | os.PathLike


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(float)
    arr = arr - np.min(arr)
    if np.max(arr) > 0:
        arr = 255 * arr / np.max(arr)
    return arr.astype(np.uint8)


def circle(
    image_length: int, row: int, col: int, radius: int, value: Any = 1, thickness: int = -1
) -> np.ndarray:
    image = np.zeros((image_length, image_length), dtype=np.uint8)
    cv.circle(image, (col, row), radius, 1, thickness=thickness)
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


@numba.njit
def filled_circle_points(r):
    size = 2 * r + 1
    arr = np.zeros((size, size), dtype=np.uint8)
    pts = np.zeros((size**2, 2), dtype=np.int32)
    perimeter = circle_points(r)
    n = len(perimeter)
    pts[:n] = perimeter

    for i in range(n):
        arr[pts[i, 0] + r, pts[i, 1] + r] = 1

    for i in range(0, 2 * r + 1):
        j = 0
        while not arr[i, j]:
            j += 1
        while arr[i, j]:
            j += 1
        if j <= r:
            while not arr[i, j]:
                pts[n, 0] = i - r
                pts[n, 1] = j - r
                n += 1
                j += 1

    return pts[:n]


@numba.njit
def circle_points(r, four_connected=False):
    # Draw a circle using the Breseham circle algorithm
    # see: https://funloop.org/post/2021-03-15-bresenham-circle-drawing-algorithm.html
    points = np.zeros((20 * r, 2), dtype=np.int32)
    x = 0
    y = -r
    # Draw the first 4 points that lie on the quadrants.
    points[:4] = np.array([[0, -r], [-r, 0], [0, r], [r, 0]], dtype=np.int32)
    x += 1
    n = 4
    while x < -y:
        # Make use of the 8-way symmetry of the circle.
        points[n : n + 8] = np.array(
            [[x, y], [y, x], [-x, y], [-y, x], [x, -y], [y, -x], [-x, -y], [-y, -x]], dtype=np.int32
        )
        n += 8
        # Test if we're currently inside or outside the circle.
        if x**2 + y**2 - r**2 <= 0:
            # We're inside so move right.
            x += 1
        else:
            # We're outside so move up.
            y += 1
            if not four_connected:
                # If we don't require 4-connected pixels then we can move diagonally.
                x += 1

    if y == -x:
        points[n : n + 4] = np.array([[x, y], [-x, -y], [-x, y], [x, -y]], dtype=np.int32)
        n += 4
    return points[:n]
