from __future__ import annotations

import math
import logging

from numpy.typing import ArrayLike
import cv2 as cv
import dask.array as da
import numpy as np
import scipy
import tqdm
import xarray as xr

from magnify import utils
import magnify.registry as registry

logger = logging.getLogger(__name__)


class ButtonFinder:
    def __init__(
        self,
        row_dist: float = 375 / 3.22,
        col_dist: float = 655 / 3.22,
        min_button_radius: int = 4,
        max_button_radius: int = 15,
        cluster_penalty: float = 10,
        roi_length: int = 61,
        progress_bar: bool = False,
        search_timesteps: list[int] | None = None,
    ):
        self.row_dist = row_dist
        self.col_dist = col_dist
        self.min_button_radius = min_button_radius
        self.max_button_radius = max_button_radius
        self.cluster_penalty = cluster_penalty
        self.roi_length = roi_length
        self.progress_bar = progress_bar
        self.search_timesteps = search_timesteps

    def __call__(self, assay: xr.Dataset) -> xr.Dataset:
        num_rows, num_cols = assay.mark_tag.shape
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

        # Store all channels and timesteps for each marker in one chunk and set marker row/col
        # sizes so each chunk ends up being at least 10MB.
        chunk_bytes = 1e7
        # Don't take into account dtype size since fg/bg bool arrays should also be 10MB.
        mark_bytes = assay.dims["channel"] * assay.dims["time"] * self.roi_length**2
        # Prioritize larger row chunks since we're more likely to want whole columns than rows.
        row_chunk_size = min(math.ceil(chunk_bytes / mark_bytes), num_rows)
        col_chunk_size = math.ceil(chunk_bytes / (mark_bytes * row_chunk_size))
        # Create the array of subimage regions.
        roi = da.empty(
            (
                num_rows,
                num_cols,
                assay.dims["channel"],
                assay.dims["time"],
                self.roi_length,
                self.roi_length,
            ),
            dtype=assay.image.dtype,
            chunks=(
                row_chunk_size,
                col_chunk_size,
                assay.dims["channel"],
                assay.dims["time"],
                self.roi_length,
                self.roi_length,
            ),
        )
        assay = assay.assign(
            roi=(
                ("mark_row", "mark_col", "channel", "time", "roi_y", "roi_x"),
                roi,
            ),
            fg=(
                ("mark_row", "mark_col", "channel", "time", "roi_y", "roi_x"),
                da.empty_like(
                    roi,
                    dtype=bool,
                ),
            ),
            bg=(
                ("mark_row", "mark_col", "channel", "time", "roi_y", "roi_x"),
                da.empty_like(
                    roi,
                    dtype=bool,
                ),
            ),
        )
        # Create the x and y coordinates arrays for each button.
        assay = assay.assign(
            x=(
                ("mark_row", "mark_col", "time"),
                np.empty((num_rows, num_cols, assay.dims["time"])),
            ),
            y=(
                ("mark_row", "mark_col", "time"),
                np.empty((num_rows, num_cols, assay.dims["time"])),
            ),
        )

        # Run the button finding algorithm for each timestep.
        for t, time in enumerate(tqdm.tqdm(assay.time, disable=not self.progress_bar)):
            # Preload all images for this timestep so we only read from disk once.
            images = assay.image.sel(time=time).compute()
            # Re-use the previous button locations if the user has specified that we should only
            # search on specific timesteps.
            do_search = t == 0 or self.search_timesteps is None or t in self.search_timesteps

            # Find button centers.
            if do_search:
                assay.x[..., t], assay.y[..., t] = self.find_centers(
                    images.sel(channel=search_channels), assay
                )
            else:
                assay.x[..., t] = assay.x[..., t - 1]
                assay.y[..., t] = assay.y[..., t - 1]

            # Extract a region around each button.
            offsets = np.empty((num_rows, num_cols, 2), dtype=int)
            roi = xr.DataArray(
                data=np.empty_like(assay.roi[:, :, :, t]), coords=assay.roi[:, :, :, t].coords
            )
            for i in range(num_rows):
                for j in range(num_cols):
                    top, bottom, left, right = utils.bounding_box(
                        round(float(assay.x[i, j, t])),
                        round(float(assay.y[i, j, t])),
                        self.roi_length,
                        assay.sizes["im_y"],
                        assay.sizes["im_x"],
                    )
                    roi[i, j] = images[:, top:bottom, left:right]
                    offsets[i, j] = [left, top]
            assay.roi[:, :, :, t] = roi

            # Compute the foreground and background masks for all buttons.
            if do_search:
                assay.fg[:, :, :, t], assay.bg[:, :, :, t] = self.find_masks(
                    roi.sel(channel=search_channels), offsets, t, assay
                )
            else:
                assay.fg[:, :, :, t] = assay.fg[:, :, :, t - 1].compute()
                assay.bg[:, :, :, t] = assay.bg[:, :, :, t - 1].compute()

        # assay = assay.stack(mark=("mark_row", "mark_col"), create_index=True).transpose(
        #     "mark", ...
        # )
        # assay = assay.set_xindex("mark_tag")

        return assay

    def find_centers(self, images: xr.DataArray, assay: xr.Dataset):
        points = np.empty((0, 2))
        min_button_dist = round(min(self.row_dist, self.col_dist) / 2)
        if min_button_dist % 2 == 0:
            # Certain opencv functions require an odd blocksize.
            min_button_dist -= 1
        for image in images:
            image = utils.to_uint8(image.to_numpy())
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
            dist_matrix = np.linalg.norm(new_points[np.newaxis] - new_points[:, np.newaxis], axis=2)
            dist_matrix[np.diag_indices(len(dist_matrix))] = np.inf
            new_points = new_points[np.min(dist_matrix, axis=1) > min_button_dist]

            if len(points) > 0:
                # Remove points too close to other points in previous channels.
                dist_matrix = np.linalg.norm(points[np.newaxis] - new_points[:, np.newaxis], axis=2)
                new_points = new_points[np.min(dist_matrix, axis=1) > min_button_dist]

            # Add the new points to the list of seen points.
            points = np.concatenate([points, new_points])

        # Split the points into x and y components.
        x = points[:, 0]
        y = points[:, 1]

        # Step 3: Cluster the points into distinct rows and columns.
        points_per_row = (assay.mark_tag != "").sum(dim="mark_col")
        points_per_col = (assay.mark_tag != "").sum(dim="mark_row")
        num_rows, num_cols = assay.sizes["mark_row"], assay.sizes["mark_col"]
        row_labels = cluster_1d(
            y,
            total_length=image.shape[0],
            num_clusters=num_rows,
            cluster_length=self.row_dist,
            ideal_num_points=points_per_row,
            penalty=self.cluster_penalty,
        )
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
        mark_y = (row_slope * col_intercepts[np.newaxis] + row_intercepts[:, np.newaxis]) / (
            1 - row_slope * col_slope
        )
        mark_x = mark_y * col_slope + col_intercepts[np.newaxis]

        return mark_x, mark_y

    def find_masks(self, roi: np.ndarray, offsets: np.ndarray, t: int, assay: xr.Dataset):
        num_rows, num_cols = roi.shape[:2]
        fg = np.empty_like(roi, dtype=bool)
        bg = np.empty_like(fg)
        for i in range(num_rows):
            for j in range(num_cols):
                # TODO: This step should occur over multiple channels.
                subimage = utils.to_uint8(roi[i, j, 0].to_numpy())

                # Find circles to refine our button estimate unless we have a blank chamber.
                circles = None
                if assay.mark_tag[i, j] != "":
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
                        minDist=self.roi_length / 2,
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
                    self.roi_length,
                    row=y_rel,
                    col=x_rel,
                    radius=self.max_button_radius,
                    value=True,
                )

                # Set the background to be the annulus around our foreground.
                bg_mask = utils.circle(
                    self.roi_length,
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

                # If enough of the button is bright then set the foreground to that bright area.
                if (fg_mask & bright_mask).sum() >= np.pi * self.min_button_radius**2:
                    fg_mask &= bright_mask

                # The background on the other hand should not be bright.
                if np.any(bg_mask & dim_mask):
                    bg_mask &= dim_mask

                fg[i, j] = fg_mask
                bg[i, j] = bg_mask

        return fg, bg

    @registry.components.register("find_buttons")
    def make(
        row_dist: float = 375 / 3.22,
        col_dist: float = 655 / 3.22,
        min_button_radius: int = 4,
        max_button_radius: int = 15,
        cluster_penalty: float = 10,
        roi_length: int = 61,
        progress_bar: bool = False,
        search_timesteps: list[int] | None = None,
    ):
        return ButtonFinder(
            row_dist=row_dist,
            col_dist=col_dist,
            min_button_radius=min_button_radius,
            cluster_penalty=cluster_penalty,
            roi_length=roi_length,
            progress_bar=progress_bar,
            search_timesteps=search_timesteps,
        )


class BeadFinder:
    def __init__(
        self,
        min_bead_radius: int = 10,
        max_bead_radius: int = 30,
        roi_length: int = 61,
        param1: int = 50,
        param2: int = 30,
    ):
        self.min_bead_radius = min_bead_radius
        self.max_bead_radius = max_bead_radius
        self.roi_length = roi_length
        self.param1 = param1
        self.param2 = param2

    def __call__(self, assay: xr.Dataset) -> xr.Dataset:
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
                roi=(
                    ("mark", "channel", "time", "roi_y", "roi_x"),
                    np.empty(
                        (
                            num_beads,
                            assay.dims["channel"],
                            assay.dims["time"],
                            self.roi_length,
                            self.roi_length,
                        ),
                        dtype=assay.image.dtype,
                    ),
                ),
                fg=(
                    ("mark", "channel", "time", "roi_y", "roi_x"),
                    np.empty(
                        (
                            num_beads,
                            assay.dims["channel"],
                            assay.dims["time"],
                            self.roi_length,
                            self.roi_length,
                        ),
                        dtype=bool,
                    ),
                ),
                bg=(
                    ("mark", "channel", "time", "roi_y", "roi_x"),
                    np.empty(
                        (
                            num_beads,
                            assay.dims["channel"],
                            assay.dims["time"],
                            self.roi_length,
                            self.roi_length,
                        ),
                        dtype=bool,
                    ),
                ),
            )
            assay = assay.assign(
                x=(
                    ["mark", "time"],
                    np.repeat(centers[:, np.newaxis, 0], assay.dims["time"], axis=1),
                ),
                y=(
                    ["mark", "time"],
                    np.repeat(centers[:, np.newaxis, 1], assay.dims["time"], axis=1),
                ),
            )

            rois = np.empty(assay.roi.shape, dtype=assay.roi.dtype)
            fgs = np.empty(assay.roi.shape, dtype=bool)
            bgs = np.empty(assay.roi.shape, dtype=bool)
            image = assay.image.to_numpy()
            # Compute the foreground and background masks for all buttons.
            for i in range(num_beads):
                # Set the subimage region for this bead.
                top, bottom, left, right = utils.bounding_box(
                    round(float(assay.x[i, 0])),
                    round(float(assay.y[i, 0])),
                    self.roi_length,
                    image.shape[-2],
                    image.shape[-1],
                )
                rois[i] = image[..., top:bottom, left:right]

                # Set the foreground of the bead to be the circle we found.
                fg_mask = utils.circle(
                    self.roi_length,
                    row=round(float(assay.y[i, 0] - top)),
                    col=round(float(assay.x[i, 0] - left)),
                    radius=round(radii[i]),
                    value=True,
                )

                # Set the background to be everything else in the region,
                # which could include other beads.
                bg_mask = ~fg_mask

                # Set the masked arrays with our computed masks.
                fgs[i] = fg_mask
                bgs[i] = bg_mask

        assay.fg[:] = fgs
        assay.bg[:] = bgs
        assay.roi[:] = rois
        return assay

    @registry.components.register("find_beads")
    def make(
        min_bead_radius: int = 10,
        max_bead_radius: int = 30,
        roi_length: int = 61,
        param1: int = 50,
        param2: int = 30,
    ):
        return BeadFinder(
            min_bead_radius=min_bead_radius,
            max_bead_radius=max_bead_radius,
            roi_length=roi_length,
            param1=param1,
            param2=param2,
        )


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
            # Just use our global estimate when we have an empty cluster.
            intercepts[i] = intercept_m * i + intercept_b

    return slope, intercepts
