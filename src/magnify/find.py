from __future__ import annotations

import logging

from numpy.typing import ArrayLike
import cv2 as cv
import dask.array as da
import numpy as np
import scipy
import xarray as xr

from magnify import utils
import magnify.registry as registry

logger = logging.getLogger(__name__)


class ButtonFinder:
    def __init__(
        self,
        row_dist: float = 126.3,
        col_dist: float = 233.2,
        min_button_radius: int = 4,
        max_button_radius: int = 15,
        cluster_penalty: float = 10,
        region_length: int = 61,
    ):
        self.row_dist = row_dist
        self.col_dist = col_dist
        self.min_button_radius = min_button_radius
        self.max_button_radius = max_button_radius
        self.cluster_penalty = cluster_penalty
        self.region_length = region_length

    def __call__(
        self,
        assay: xr.Dataset,
        row_dist: float = 126.3,
        col_dist: float = 233.2,
        min_button_radius: int = 4,
        max_button_radius: int = 15,
        cluster_penalty: float = 10,
        region_length: int = 61,
    ) -> xr.Dataset:
        self.row_dist = row_dist
        self.col_dist = col_dist
        self.min_button_radius = min_button_radius
        self.max_button_radius = max_button_radius
        self.cluster_penalty = cluster_penalty
        self.region_length = region_length

        num_rows, num_cols = assay.id.shape
        if isinstance(assay.search_channel, str):
            if assay.search_channel in assay.channel:
                search_channels = [assay.search_channel]
            elif assay.search_channel == "all":
                search_channels = assay.channel
            else:
                raise ValueError(f"{assay.search_channel} is not a channel name.")
        else:
            # We're searching across multiple channels.
            search_channels = assay.search_channel
        min_button_dist = round(min(self.row_dist, self.col_dist) / 2)
        if min_button_dist % 2 == 0:
            # Certain opencv functions require an odd blocksize.
            min_button_dist -= 1

        # Create the array of subimage regions.
        regions = da.empty(
            (
                num_rows,
                num_cols,
                assay.dims["channel"],
                assay.dims["time"],
                self.region_length,
                self.region_length,
            ),
            dtype=assay.image.dtype,
            chunks=(
                1,
                1,
                assay.dims["channel"],
                assay.dims["time"],
                self.region_length,
                self.region_length,
            ),
        )
        assay = assay.assign(
            region=(
                ("marker_row", "marker_col", "channel", "time", "row", "col"),
                regions,
            ),
            fg=(
                ("marker_row", "marker_col", "channel", "time", "row", "col"),
                da.empty_like(
                    regions,
                    dtype=bool,
                ),
            ),
            bg=(
                ("marker_row", "marker_col", "channel", "time", "row", "col"),
                da.empty_like(
                    regions,
                    dtype=bool,
                ),
            ),
        )
        # Create the x and y coordinates arrays for each button.
        assay = assay.assign_coords(
            x=(
                ["marker_row", "marker_col", "time"],
                np.empty((num_rows, num_cols, assay.dims["time"])),
            ),
            y=(
                ["marker_row", "marker_col", "time"],
                np.empty((num_rows, num_cols, assay.dims["time"])),
            ),
        )

        # Run the button finding algorithm for each timestep.
        for t, time in enumerate(assay.time):
            # Preload all images for this timnepoint so we only read from disk once.
            images = assay.image.sel(time=time).to_numpy()
            points = np.empty((0, 2))
            for channel in search_channels:
                c = np.where(assay.channel == channel)[0][0]
                image = utils.to_uint8(images[c])
                # Step 1: Find an imperfect button mask by thresholding.
                mask = cv.adaptiveThreshold(
                    image,
                    maxValue=255,
                    adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                    thresholdType=cv.THRESH_BINARY,
                    blockSize=min_button_dist,
                    C=-1,
                )
                # Step 2: Find connected components and filter out points.
                _, _, stats, new_points = cv.connectedComponentsWithStats(mask, connectivity=4)

                # Ignore the background point.
                new_points = new_points[1:]
                stats = stats[1:]

                # Exclude large and small blobs.
                new_points = new_points[
                    (stats[:, cv.CC_STAT_HEIGHT] <= 2 * self.max_button_radius)
                    & (stats[:, cv.CC_STAT_WIDTH] <= 2 * self.max_button_radius)
                    & (stats[:, cv.CC_STAT_HEIGHT] >= 2 * self.min_button_radius)
                    & (stats[:, cv.CC_STAT_WIDTH] >= 2 * self.min_button_radius)
                ]
                # Remove points too close to other points in this channel.
                dist_matrix = np.linalg.norm(
                    new_points[np.newaxis] - new_points[:, np.newaxis], axis=2
                )
                dist_matrix[np.diag_indices(len(dist_matrix))] = np.inf
                new_points = new_points[np.min(dist_matrix, axis=1) > min_button_dist]

                if len(points) > 0:
                    # Remove points too close to other points in previous channels.
                    dist_matrix = np.linalg.norm(
                        points[np.newaxis] - new_points[:, np.newaxis], axis=2
                    )
                    new_points = new_points[np.min(dist_matrix, axis=1) > min_button_dist]

                # Add the new points to the list of seen points.
                points = np.concatenate([points, new_points])

            # Split the points into x and y components.
            x = points[:, 0]
            y = points[:, 1]

            # Step 3: Cluster the points into distinct rows and columns.
            # The number of buttons we expect to see per row/col can vary if we have blank buttons.
            points_per_row = (assay.id != "").sum(dim="marker_col")
            row_labels = cluster_1d(
                y,
                total_length=image.shape[0],
                num_clusters=num_rows,
                cluster_length=self.row_dist,
                ideal_num_points=points_per_row,
                penalty=self.cluster_penalty,
            )
            points_per_col = (assay.id != "").sum(dim="marker_row")
            col_labels = cluster_1d(
                x,
                total_length=image.shape[1],
                num_clusters=num_cols,
                cluster_length=self.col_dist,
                ideal_num_points=points_per_col,
                penalty=self.cluster_penalty,
            )

            # Exclude boundary points that didn't fall into clusters.
            in_cluster = (row_labels >= 0) & (col_labels >= 0)
            x, y = x[in_cluster], y[in_cluster]
            col_labels, row_labels = col_labels[in_cluster], row_labels[in_cluster]

            # Step 4: Draw lines through each cluster.
            row_slope, row_intercepts = regress_clusters(
                x, y, labels=row_labels, num_clusters=num_rows, ideal_num_points=points_per_row
            )
            # We treat column indices as y and row indices as x to avoid near-infinite slopes.
            col_slope, col_intercepts = regress_clusters(
                y,
                x,
                labels=col_labels,
                num_clusters=num_cols,
                ideal_num_points=points_per_col,
            )

            # Step 5: Set button locations as the intersection of each line pair.
            assay.y[..., t] = (
                row_slope * col_intercepts[np.newaxis] + row_intercepts[:, np.newaxis]
            ) / (1 - row_slope * col_slope)
            assay.x[..., t] = assay.y[..., t] * col_slope + col_intercepts[np.newaxis]

            # Step 6: Extract a region around each button.
            offsets = np.empty((num_rows, num_cols, 2), dtype=int)
            regions = np.empty(
                (num_rows, num_cols, assay.dims["channel"], self.region_length, self.region_length),
                dtype=assay.region.dtype,
            )
            for i in range(num_rows):
                for j in range(num_cols):
                    top, bottom, left, right = utils.bounding_box(
                        round(float(assay.x[i, j, t])),
                        round(float(assay.y[i, j, t])),
                        self.region_length,
                        assay.dims["im_row"],
                        assay.dims["im_col"],
                    )
                    regions[i, j] = images[:, top:bottom, left:right]
                    offsets[i, j] = [left, top]
            assay.region[:, :, :, t] = regions

            # Step 7: Compute the foreground and background masks for all buttons.
            fg = np.empty(
                (num_rows, num_cols, assay.dims["channel"], self.region_length, self.region_length),
                dtype=bool,
            )
            bg = np.empty_like(fg)
            for i in range(num_rows):
                for j in range(num_cols):
                    # TODO: This step should occur over multiple channels.
                    c = np.where(assay.channel == search_channels[0])[0][0]
                    subimage = utils.to_uint8(regions[i, j, c])
                    # Filter the subimage to smooth edges and remove noise.
                    filtered = cv.bilateralFilter(
                        subimage,
                        d=9,
                        sigmaColor=75,
                        sigmaSpace=75,
                        borderType=cv.BORDER_DEFAULT,
                    )

                    # Find any circles in the subimage.
                    circles = cv.HoughCircles(
                        filtered,
                        method=cv.HOUGH_GRADIENT,
                        dp=1,
                        minDist=50,
                        param1=20,
                        param2=5,
                        minRadius=self.min_button_radius,
                        maxRadius=self.max_button_radius,
                    )

                    # Update our estimate of the button position if we found some circles.
                    if circles is not None:
                        circles = circles[0, :, :2]
                        point = np.array([assay.x[i, j, t], assay.y[i, j, t]]) - offsets[i, j]
                        # Use the circle center closest to our previous estimate of the button.
                        closest_idx = np.argmin(np.linalg.norm(circles - point, axis=1))
                        assay.x[i, j, t], assay.y[i, j, t] = circles[closest_idx] + offsets[i, j]

                    x_rel = round(float(assay.x[i, j, t])) - offsets[i, j, 0]
                    y_rel = round(float(assay.y[i, j, t])) - offsets[i, j, 1]

                    # Set the foreground (the button) to be a circle of fixed radius.
                    fg_mask = utils.circle(
                        self.region_length,
                        row=y_rel,
                        col=x_rel,
                        radius=self.max_button_radius,
                        value=True,
                    )

                    # Set the background to be the annulus around our foreground.
                    bg_mask = utils.circle(
                        self.region_length,
                        row=y_rel,
                        col=x_rel,
                        radius=2 * self.max_button_radius,
                        value=True,
                    )
                    bg_mask &= ~fg_mask

                    # Refine the foreground & background by finding areas that are bright and dim.
                    _, bright_mask = cv.threshold(
                        subimage, thresh=0, maxval=1, type=cv.THRESH_BINARY + cv.THRESH_OTSU
                    )
                    dim_mask = ~cv.dilate(
                        bright_mask, np.ones((self.max_button_radius, self.max_button_radius))
                    )
                    bright_mask = bright_mask.astype(bool)
                    dim_mask = dim_mask.astype(bool)

                    # If part of the button is bright then set the foreground to that bright area.
                    if np.any(fg_mask & bright_mask):
                        fg_mask &= bright_mask

                    # The background on the other hand should not be bright.
                    if np.any(bg_mask & dim_mask):
                        bg_mask &= dim_mask

                    fg[i, j] = fg_mask
                    bg[i, j] = bg_mask

            assay.fg[:, :, :, t] = fg
            assay.bg[:, :, :, t] = bg

        return assay

    @registry.components.register("find_buttons")
    def make():
        return ButtonFinder()


class BeadFinder:
    def __init__(
        self,
        min_bead_radius: int = 10,
        max_bead_radius: int = 30,
        region_length: int = 61,
        param1: int = 50,
        param2: int = 30,
    ):
        self.min_bead_radius = min_bead_radius
        self.max_bead_radius = max_bead_radius
        self.region_length = region_length
        self.param1 = param1
        self.param2 = param2

    def __call__(
        self,
        assay: xr.Dataset,
        min_bead_radius: int = 10,
        max_bead_radius: int = 30,
        region_length: int = 61,
        param1: int = 50,
        param2: int = 30,
    ) -> xr.Dataset:
        self.min_bead_radius = min_bead_radius
        self.max_bead_radius = max_bead_radius
        self.region_length = region_length
        self.param1 = param1
        self.param2 = param2
        if isinstance(assay.search_channel, str):
            if assay.search_channel in assay.channel:
                search_channels = [assay.search_channel]
            elif assay.search_channel == "all":
                search_channels = assay.channel
            else:
                raise ValueError(f"{assay.search_channel} is not a channel name.")
        else:
            # We're searching across multiple channels.
            search_channels = assay.search_channel

        centers = np.empty((0, 2))
        radii = np.empty((0))
        for t in assay.time:
            for search_channel in search_channels:
                image = utils.to_uint8(assay.image.sel(channel=search_channel, time=t).to_numpy())
                # Filter the subimage to smooth edges and remove noise.
                filtered = cv.bilateralFilter(
                    image,
                    d=9,
                    sigmaColor=75,
                    sigmaSpace=75,
                    borderType=cv.BORDER_DEFAULT,
                )

                # Find any circles in the image.
                circles = cv.HoughCircles(
                    filtered,
                    method=cv.HOUGH_GRADIENT,
                    dp=1,
                    minDist=2 * self.min_bead_radius,
                    param1=self.param1,
                    param2=self.param2,
                    minRadius=self.min_bead_radius,
                    maxRadius=self.max_bead_radius,
                )

                if circles is not None:
                    circles = circles[0]
                    # Save circle center locations.
                    c = circles[:, :2]
                    if len(centers) > 0:
                        # Remove centers too close to those we already found in another channel or time.
                        dist_matrix = np.linalg.norm(c[np.newaxis] - centers[:, np.newaxis], axis=2)
                        valid = np.min(dist_matrix, axis=0) > self.min_bead_radius
                    else:
                        valid = np.ones(len(circles), dtype=bool)

                    centers = np.concatenate([centers, c[valid]])
                    radii = np.concatenate([radii, circles[valid, 2]])

        # Update the assay object with the beads we found.
        if len(centers) > 0:
            num_beads = len(centers)
            # Create the array of subimage regions.
            assay = assay.assign(
                region=(
                    ("marker", "channel", "time", "row", "col"),
                    np.empty(
                        (
                            num_beads,
                            assay.dims["channel"],
                            assay.dims["time"],
                            self.region_length,
                            self.region_length,
                        ),
                        dtype=assay.image.dtype,
                    ),
                ),
                fg=(
                    ("marker", "channel", "time", "row", "col"),
                    np.empty(
                        (
                            num_beads,
                            assay.dims["channel"],
                            assay.dims["time"],
                            self.region_length,
                            self.region_length,
                        ),
                        dtype=bool,
                    ),
                ),
                bg=(
                    ("marker", "channel", "time", "row", "col"),
                    np.empty(
                        (
                            num_beads,
                            assay.dims["channel"],
                            assay.dims["time"],
                            self.region_length,
                            self.region_length,
                        ),
                        dtype=bool,
                    ),
                ),
            )
            assay = assay.assign_coords(
                x=(
                    ["marker", "time"],
                    np.repeat(centers[:, np.newaxis, 0], assay.dims["time"], axis=1),
                ),
                y=(
                    ["marker", "time"],
                    np.repeat(centers[:, np.newaxis, 1], assay.dims["time"], axis=1),
                ),
            )

            # Compute the foreground and background masks for all buttons.
            for i in range(num_beads):
                # Set the subimage region for this bead.
                top, bottom, left, right = utils.bounding_box(
                    round(float(assay.x[i, 0])),
                    round(float(assay.y[i, 0])),
                    self.region_length,
                    image.shape[-2],
                    image.shape[-1],
                )
                assay.region[i] = assay.image.sel(
                    im_row=slice(top, bottom), im_col=slice(left, right)
                )

                # Set the foreground of the bead to be the circle we found.
                fg_mask = utils.circle(
                    self.region_length,
                    row=round(float(assay.y[i, 0] - top)),
                    col=round(float(assay.x[i, 0] - left)),
                    radius=round(radii[i]),
                    value=True,
                )

                # Set the background to be everything else in the region,
                # which could include other beads.
                bg_mask = ~fg_mask

                # Set the masked arrays with our computed masks.
                assay.fg[i] = fg_mask
                assay.bg[i] = bg_mask

        return assay

    @registry.components.register("find_beads")
    def make():
        return BeadFinder()


def cluster_1d(
    points: np.ndarray,
    total_length: int,
    num_clusters: int,
    cluster_length: float,
    ideal_num_points: np.ndarray,
    penalty: float,
) -> np.ndarray:
    # Find the best clustering using the accumulate ragged array trick.
    # See: https://vladfeinberg.com/2021/01/07/vectorizing-ragged-arrays.html
    permutation = np.argsort(points)
    points = points[permutation]

    def cost(offset: int) -> tuple[float, np.ndarray]:
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
        # Fewer points in a cluster means noisier variances so adjust variance contributions.
        cost *= np.sqrt(ideal_num_points)
        # Penalize clusters for having too few or too many points.
        cost = cost + penalty * (ideal_num_points - num_points) ** 2
        return np.sum(cost), spans

    spans = min(
        (cost(i) for i in range(total_length - round(num_clusters * cluster_length))),
        key=lambda x: x[0],
    )[1]

    # Label each point with its cluster, label points outside clusters as -1.
    labels = -np.ones_like(points, dtype=int)
    labels[spans[0] : spans[-1]] = np.repeat(np.arange(num_clusters), spans[1:] - spans[:-1])

    # Return the labels based on the original order of the points.
    return labels[np.argsort(permutation)]


def regress_clusters(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
    ideal_num_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Find the best line per-cluster.
    slopes = np.full(num_clusters, np.nan)
    intercepts = np.full(num_clusters, np.nan)
    cluster_points = [(x[labels == i], y[labels == i]) for i in range(num_clusters)]
    for i, (x, y) in enumerate(cluster_points):
        # Only regress on multi-point clusters.
        if len(x) > 1:
            slopes[i], intercepts[i], _, _, _ = scipy.stats.linregress(x, y)
        elif (i == 0 or i == num_clusters - 1) and ideal_num_points[i] >= 2:
            logger.warning(
                "Boundary cluster has fewer than 2 points."
                "The chip is unlikely to be segmented correctly."
            )

    # Recompute the intercepts using the median slope.
    slope = np.nanmedian(slopes)
    for i, (x, y) in enumerate(cluster_points):
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
    for i, (x, y) in enumerate(cluster_points):
        if ideal_num_points[i] != 0 and not_nan[i]:
            weight = min(len(x), ideal_num_points[i]) / ideal_num_points[i]
            intercepts[i] = weight * intercepts[i] + (1 - weight) * (intercept_m * i + intercept_b)
        else:
            # Just use our global estimate when we have an cluster.
            intercepts[i] = intercept_m * i + intercept_b

    return slope, intercepts
