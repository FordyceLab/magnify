import logging

from numpy.typing import ArrayLike
import cv2 as cv
import numpy as np
import scipy

from magnify import utils
from magnify.assay import Assay

logger = logging.getLogger(__name__)


def find_buttons(
    images: np.ndarray,
    channels: ArrayLike,
    num_rows: int = 46,
    num_cols: int = 24,
    row_dist: float = 126.3,
    col_dist: float = 233.2,
    min_button_radius: int = 4,
    max_button_radius: int = 15,
    cluster_penalty: float = 10,
    search_on: str = "egfp",
) -> Assay:
    channels = np.array(channels)
    idx = np.where(channels == search_on)[0][0]
    image = utils.to_uint8(images[idx])
    min_button_dist = round(min(row_dist, col_dist) / 2)
    if min_button_dist % 2 == 0:
        min_button_dist -= 1

    # Step 1: Find an imperfect button mask by thresholding.
    mask = cv.adaptiveThreshold(
        image,
        maxValue=255,
        adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv.THRESH_BINARY,
        blockSize=min_button_dist,
        C=-1,
    )
    # Remove large blobs.
    mask = cv.morphologyEx(
        mask,
        op=cv.MORPH_TOPHAT,
        kernel=np.ones((2 * max_button_radius, 2 * max_button_radius)),
    )
    # Remove small blobs.
    mask = cv.erode(
        mask, kernel=np.ones((2 * min_button_radius, 2 * min_button_radius))
    )

    # Step 2: Get all connected components and filter out ones too close to each other.
    points = cv.connectedComponentsWithStats(mask, connectivity=4)[3]
    # Ignore the background point and change indexing to be in row-col order.
    points = points[1:, ::-1]
    dist_matrix = np.linalg.norm(points[np.newaxis] - points[:, np.newaxis], axis=2)
    dist_matrix[np.diag_indices(len(dist_matrix))] = np.inf
    points = points[np.min(dist_matrix, axis=0) > min_button_dist]

    # Step 3: Cluster the points into distinct rows and columns.
    row_labels = cluster_1d(
        points[:, 0],
        total_length=image.shape[0],
        num_clusters=num_rows,
        cluster_length=row_dist,
        ideal_num_points=num_cols,
        penalty=cluster_penalty,
    )
    col_labels = cluster_1d(
        points[:, 1],
        total_length=image.shape[1],
        num_clusters=num_cols,
        cluster_length=col_dist,
        ideal_num_points=num_rows,
        penalty=cluster_penalty,
    )

    # Exclude boundary points that didn't fall into clusters.
    in_cluster = (row_labels >= 0) & (col_labels >= 0)
    points = points[in_cluster]
    col_labels = col_labels[in_cluster]
    row_labels = row_labels[in_cluster]

    # Step 4: Draw lines through each cluster.
    row_slope, row_intercepts = regress_clusters(
        points, labels=row_labels, num_clusters=num_rows, ideal_num_points=num_cols
    )
    # We treat column indices as y and row indices as x to avoid near-infinite slopes.
    col_slope, col_intercepts = regress_clusters(
        points[:, ::-1],
        labels=col_labels,
        num_clusters=num_cols,
        ideal_num_points=num_rows,
    )

    # Step 5: Set button locations as the intersection of each line pair.
    button_pos = np.empty((num_rows, num_cols, 2))
    button_pos[:, :, 0] = (
        row_slope * col_intercepts[np.newaxis] + row_intercepts[:, np.newaxis]
    ) / (1 - row_slope * col_slope)
    button_pos[:, :, 1] = button_pos[:, :, 0] * col_slope + col_intercepts[np.newaxis]

    assay = Assay()
    assay.type = "chip"
    assay.channels = channels
    assay.images = images
    assay.centers = button_pos
    assay.valid = np.ones((len(channels), num_rows, num_cols), dtype=bool)

    return assay


def cluster_1d(
    points: np.ndarray,
    total_length: int,
    num_clusters: int,
    cluster_length: float,
    ideal_num_points: int,
    penalty: float,
) -> np.ndarray:
    # Find the best clustering using the accumulate ragged array trick.
    # See: https://vladfeinberg.com/2021/01/07/vectorizing-ragged-arrays.html
    permutation = np.argsort(points)
    points = points[permutation]

    def cost(offset):
        # Compute the boundaries and center of each cluster.
        boundaries = np.arange(num_clusters + 1) * cluster_length + offset
        centers = (boundaries[1:] + boundaries[:-1]) / 2
        # Find start/end indexes and number of points in each cluster.
        spans = np.searchsorted(points, boundaries)
        num_points = spans[1:] - spans[:-1]
        # Compute the squared distance of each point that falls in a cluster to its cluster center.
        dists = (points[spans[0] : spans[-1]] - np.repeat(centers, num_points)) ** 2
        # Penalize clusters for high variance.
        cost = np.insert(np.cumsum(dists), 0, 0)
        cost = np.diff(cost[spans - spans[0]])
        cost[num_points > 0] /= num_points[num_points > 0]
        # Set empty clusters to the maximum variance.
        cost[num_points == 0] = np.max(cost)
        # Penalize clusters for having too few or too many points.
        cost += penalty * (ideal_num_points - num_points) ** 2
        return np.sum(cost), spans

    spans = min(
        (cost(i) for i in range(total_length - round(num_clusters * cluster_length))),
        key=lambda x: x[0],
    )[1]

    # Label each point with its cluster, label points outside clusters as -1.
    labels = -np.ones_like(points, dtype=int)
    labels[spans[0] : spans[-1]] = np.repeat(
        np.arange(num_clusters), spans[1:] - spans[:-1]
    )

    # Return the labels based on the original order of the points.
    return labels[np.argsort(permutation)]


def regress_clusters(
    points: np.ndarray, labels: np.ndarray, num_clusters: int, ideal_num_points: int
) -> tuple[np.ndarray, np.ndarray]:
    # Find the best line per-cluster.
    slopes = np.full(num_clusters, np.nan)
    intercepts = np.full(num_clusters, np.nan)
    points = [points[labels == i].T for i in range(num_clusters)]
    for i, (y, x) in enumerate(points):
        # Only regress on multi-point clusters.
        if len(x) > 1:
            slopes[i], intercepts[i], _, _, _ = scipy.stats.linregress(x, y)
        elif i == 0 or i == num_clusters - 1:
            logger.warning(
                f"Boundary cluster only has less than 2 points."
                "The chip is unlikely to be segmented correctly."
            )

    # Recompute the intercepts using the median slope.
    slope = np.nanmedian(slopes)
    for i, (y, x) in enumerate(points):
        if len(x) > 0:
            intercepts[i] = np.median(y - slope * x)

    # Globally estimate where intercepts are, assuming they're evenly spaced.
    not_nan = ~np.isnan(intercepts)
    label_idxs = np.arange(num_clusters)
    intercept_m, intercept_b, _, _, _ = scipy.stats.linregress(
        label_idxs[not_nan], intercepts[not_nan]
    )
    # Re-estimate intercepts using a weighted mean of global and local estimates.
    # This reduces outlier effects while still allowing uneven intercepts from image stitching.
    for i, (y, x) in enumerate(points):
        weight = min(len(x), ideal_num_points) / ideal_num_points
        intercepts[i] = weight * intercepts[i] + (1 - weight) * (
            intercept_m * i + intercept_b
        )

    return slope, intercepts
