import inspect
import math
import os
import re
from collections.abc import Callable, Iterable
from typing import Annotated, Any

import cv2 as cv
import napari
import napari.types
import numba
import numpy as np
from numba import prange

from magnify.plot.vis import InteractiveUI

PathLike = str | bytes | os.PathLike


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(float)
    arr = arr - np.min(arr)
    if np.max(arr) > 0:
        arr = 255 * arr / np.max(arr)
    return arr.astype(np.uint8)


def circle(
    image_shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
    value: Any = 1,
    thickness: int = -1,
) -> np.ndarray:
    image = np.zeros(image_shape, dtype=np.uint8)
    cv.circle(image, (center[1], center[0]), radius, 1, thickness=thickness)
    image = image.astype(type(value)) * value
    return image


def annulus(
    image_shape: tuple[int, int],
    center: tuple[int, int],
    outer_radius: int,
    inner_radius: int,
    value: Any = 1.0,
) -> np.ndarray:
    outer_circle = circle(image_shape, center, outer_radius, value)
    inner_circle = circle(image_shape, center, inner_radius, value)
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


def find_circles(
    img: np.ndarray,
    low_edge_quantile: float,
    high_edge_quantile: float,
    grid_length: int,
    num_iter: int,
    min_radius: int,
    max_radius: int,
    min_roundness: float,
    min_dist: int,
    gui: InteractiveUI | None,
) -> tuple[np.ndarray, np.ndarray]:
    # TODO: Make this functions nicer.
    # Step 1: Denoise the image for more accurate edge finding.
    img = cv.GaussianBlur(img, (5, 5), 0)

    # Step 2: Find edges from image gradients.
    dx = cv.Scharr(img, ddepth=cv.CV_32F, dx=1, dy=0)
    dy = cv.Scharr(img, ddepth=cv.CV_32F, dx=0, dy=1)
    grad = np.sqrt(dx**2 + dy**2)

    def compute_edges(
        low_edge_quantile: Annotated[float, {"max": 1.0, "step": 0.001}] = low_edge_quantile,
        high_edge_quantile: Annotated[float, {"max": 1.0, "step": 0.001}] = high_edge_quantile,
    ) -> list[napari.types.LayerDataTuple]:
        low_thresh = np.quantile(grad, low_edge_quantile)
        high_thresh = np.quantile(grad, high_edge_quantile)
        edges = cv.Canny(
            dx.astype(np.int16),
            dy.astype(np.int16),
            threshold1=low_thresh,
            threshold2=high_thresh,
            L2gradient=True,
        )
        return [(img, {"name": "Image"}), (edges, {"name": "Edges", "blending": "additive"})]

    if gui is not None:
        edges = gui.run_widget(compute_edges, auto_call=True)[1][0]
    else:
        edges = compute_edges()[1][0]

    edges[edges != 0] = 1

    # Step 3: Use edges to find candidate circles.
    all_circles = candidate_circles(edges, grid_length, num_iter)

    retval = [0, 0]

    def filter_circles(
        min_radius: int = min_radius,
        max_radius: int = max_radius,
        min_roundness: Annotated[float, {"max": 1.0}] = min_roundness,
        min_dist: int = min_dist,
    ) -> list[napari.types.LayerDataTuple]:
        # Step 4: Filter circles based on size and position.
        # Remove circles that are too small or large.
        circles = all_circles[(all_circles[:, 2] >= min_radius) & (all_circles[:, 2] <= max_radius)]
        # Round circle coordinates since we'll only be considering whole pixels.
        circles = np.round(circles).astype(np.int32)
        # Remove circles that that are completely off the image.
        circles = circles[
            (circles[:, 0] + circles[:, 2] >= 0)
            & (circles[:, 1] + circles[:, 2] >= 0)
            & (circles[:, 0] - circles[:, 2] < img.shape[0])
            & (circles[:, 1] - circles[:, 2] < img.shape[1])
        ]

        # Step 5: Filter circles with low number of edges on their circumference
        # or whose gradients don't point toward the center.
        thetas = np.arctan2(dy, dx)
        # Pad the gradient angles and edges to avoid boundary cases.
        pad = 2 * max_radius
        thetas = np.pad(thetas, pad)
        padded_edges = np.pad(edges, pad)
        # Adjust circle coordinates to account for padding.
        circles[:, :2] += pad
        # Sort circles by radius.
        order = np.argsort(circles[:, 2])
        circles = circles[order]

        start = 0
        scores = []
        for radius in range(min_radius, max_radius + 1):
            perimeter_coords = circle_points(radius)
            end = np.searchsorted(circles[:, 2], radius + 1)
            s = mean_grad(thetas, padded_edges, circles[start:end, :2], perimeter_coords)
            scores.append(s / len(perimeter_coords))
            start = end
        circles[:, :2] -= pad
        scores = np.concatenate(scores)
        circles = circles[scores >= min_roundness]
        scores = scores[scores >= min_roundness]

        # Step 6: Remove duplicate circles that are too close to each other.
        perm = np.argsort(-scores)
        circles, scores = circles[perm], scores[perm]
        if min_dist > 0:
            valid = filter_neighbors(circles, min_dist)
            circles, scores = circles[valid], scores[valid]

        retval[0], retval[1] = circles, scores
        return [
            (img, {"name": "Image"}),
            (
                circles[:, :2],
                {
                    "name": "Circles",
                    "size": 2 * circles[:, 2],
                    "border_color": "white",
                    "face_color": [0] * 4,
                    "blending": "additive",
                },
                "points",
            ),
        ]

    if gui is not None:
        gui.run_widget(filter_circles, auto_call=True, last=True)
    else:
        filter_circles()

    return tuple(retval)


@numba.njit(parallel=True)
def mean_grad(thetas, edges, circles, perimeter_coords):
    # TODO: Make this function nicer.
    means = np.empty(len(circles), dtype=np.float32)
    c = np.arctan2(perimeter_coords[:, 0], perimeter_coords[:, 1])
    for i in prange(len(circles)):
        row = perimeter_coords[:, 0] + circles[i, 0]
        col = perimeter_coords[:, 1] + circles[i, 1]
        m = 0
        for j in prange(len(row)):
            t = thetas[row[j], col[j]]
            e = edges[row[j], col[j]]
            if e > 0:
                d = np.abs(t - c[j])
                if d > np.pi:
                    d = d - np.pi
                m += 4 * np.abs(d - np.pi / 2) / np.pi - 1
        means[i] = m

    return means


@numba.njit  # (parallel=True)
def filter_neighbors(circles, min_dist):
    # TODO: Make this function nicer.
    coords = circle_points(min_dist, four_connected=True)

    pad = 2 * min_dist + 1
    arr = -np.ones((circles[:, 0].max() + 2 * pad, circles[:, 1].max() + 2 * pad), dtype=np.int32)
    valid = np.ones(len(circles), dtype=np.bool_)
    for i in range(len(circles)):
        for j in range(len(coords)):
            row = coords[j, 0] + circles[i, 0] + pad
            col = coords[j, 1] + circles[i, 1] + pad
            v = arr[row, col]
            if v != -1:
                valid[i] = False
                break
        if not valid[i]:
            continue
        for j in range(len(coords)):
            row = coords[j, 0] + circles[i, 0] + pad
            col = coords[j, 1] + circles[i, 1] + pad
            arr[row, col] = i

    return valid


@numba.njit(parallel=True)
def candidate_circles(edges, grid_length, num_iter):
    # Find the coordinates of all edges.
    coords = np.column_stack(np.where(edges))

    # Create a coarse grid of the image for fast neighbor lookup.
    grid_coords, grid_starts, grid_counts = grid_array(edges, grid_length)

    # Generate the candidate circles by randomly sampling three nearby points
    # and finding the circle that passes through these points.
    circles = np.empty((num_iter, 3), dtype=np.float32)
    for i in prange(num_iter):
        # Randomly select a point from all edges.
        p0 = coords[np.random.choice(len(coords))]
        p0_grid = p0 // grid_length

        # Randomly choose the next two points from the same grid cell as p0 and center the
        # entire coordinate system on p0 to make calculations easier.
        row = p0_grid[0]
        col = p0_grid[1]
        idx = grid_starts[row, col] + np.random.choice(grid_counts[row, col])
        p1 = grid_coords[idx] - p0
        idx = grid_starts[row, col] + np.random.choice(grid_counts[row, col])
        p2 = grid_coords[idx] - p0

        # The circle that passes through all three points will have a center at the intersection of
        # the perpendicular bisectors of p0-p1 and p0-p2: https://en.wikipedia.org/wiki/Circumcircle
        # Find the p0-p1 and p0-p2 midpoints.
        mid1 = np.float32(0.5) * p1
        mid2 = np.float32(0.5) * p2
        # Find the slope and intercept of the perpendicular bisectors. Adding a small value to the
        # denominator accounts for vertical slopes.
        eps = np.float32(1e-20)
        m1 = -p1[1] / (p1[0] + eps)
        m2 = -p2[1] / (p2[0] + eps)
        b1 = mid1[0] - m1 * mid1[1]
        b2 = mid2[0] - m2 * mid2[1]
        # Find the intersection of the two bisectors. If the two bisectors are parallel then
        # the center coordinates will be large and the circle will get filtered out later.
        circles[i, 1] = (b1 - b2) / (m2 - m1 + eps)
        circles[i, 0] = m1 * circles[i, 1] + b1
        # Find the radius of the circle with the knowledge that p0 i.e. (0, 0) is on the circle.
        circles[i, 2] = np.sqrt(circles[i, 0] ** 2 + circles[i, 1] ** 2)
        # Recenter the circle coordinates back to their original position.
        circles[i, :2] = circles[i, :2] + p0

    return circles


@numba.njit
def grid_array(arr, grid_length):
    num_rows = math.ceil(arr.shape[0] / grid_length)
    num_cols = math.ceil(arr.shape[1] / grid_length)
    grid_counts = np.empty((num_rows, num_cols), dtype=np.int64)
    for i in range(num_rows):
        for j in range(num_cols):
            grid_counts[i, j] = arr[
                i * grid_length : (i + 1) * grid_length,
                j * grid_length : (j + 1) * grid_length,
            ].sum()

    # The number of edges in each grid cell is variable in length so we'll store
    # their coordinates as a flat array and keep track of the starting indices for each grid cell.
    grid_starts = np.empty_like(grid_counts)
    grid_coords = np.empty((grid_counts.sum(), 2), dtype=np.int32)
    n = 0
    for i in range(num_rows):
        for j in range(num_cols):
            r, c = np.where(
                arr[
                    i * grid_length : (i + 1) * grid_length,
                    j * grid_length : (j + 1) * grid_length,
                ]
            )
            grid_starts[i, j] = n
            grid_coords[n : n + len(r), 0] = r + i * grid_length
            grid_coords[n : n + len(r), 1] = c + j * grid_length
            n += len(r)

    return grid_coords, grid_starts, grid_counts


@numba.njit
def circle_labels(circles, num_rows, num_cols):
    labels = -1 * np.ones((num_rows, num_cols), dtype=np.int32)

    for i in range(len(circles)):
        pts = filled_circle_points(circles[i, 2])
        pts += circles[i, :2]
        for j in prange(len(pts)):
            r, c = pts[j]
            if r >= 0 and r < num_rows and c >= 0 and c < num_cols:
                if labels[r, c] != -1:
                    labels[r, c] = -2
                else:
                    labels[r, c] = i

    return labels


@numba.njit
def filled_circle_points(r):
    size = 2 * r + 1
    arr = np.zeros((size, size), dtype=np.uint8)
    pts = np.zeros((size**2, 2), dtype=np.int32)
    perimeter = circle_points(r)
    n = len(perimeter)
    pts[:n] = perimeter

    # First fill the circumference.
    for i in range(n):
        arr[pts[i, 0] + r, pts[i, 1] + r] = 1

    # Fill the interior of the circle one row at a time.
    for i in range(0, 2 * r + 1):
        j = 0
        # Iterate until we hit the circle's circumference.
        while not arr[i, j]:
            j += 1
        # Iterate just past the circle's circumference.
        while arr[i, j]:
            j += 1
        # Skip when the row only has circumference and no interior.
        if j <= r:
            # We are in the interior of the circle so fill in the pts array until
            # we hit the circumference again
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
            [[x, y], [y, x], [-x, y], [-y, x], [x, -y], [y, -x], [-x, -y], [-y, -x]],
            dtype=np.int32,
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
