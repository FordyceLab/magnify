from __future__ import annotations

import math

import cv2 as cv
import dask.array as da
import numba
import numpy as np
import scipy
import tqdm
import xarray as xr
from numba import prange

import magnify.registry as registry
from magnify import logger, utils


class ButtonFinder:
    def __init__(
        self,
        row_dist: float = 375 / 3.22,
        col_dist: float = 655 / 3.22,
        min_button_radius: int = 4,
        max_button_radius: int = 15,
        low_edge_quantile: float = 0.1,
        high_edge_quantile: float = 0.9,
        num_iter: int = 5000000,
        min_roundness: float = 0.2,
        cluster_penalty: float = 50,
        roi_length: int = 61,
        progress_bar: bool = False,
        search_timestep: list[int] | None = None,
        search_channel: str | list[str] | None = None,
    ):
        self.row_dist = row_dist
        self.col_dist = col_dist
        self.min_button_radius = min_button_radius
        self.max_button_radius = max_button_radius
        self.low_edge_quantile = low_edge_quantile
        self.high_edge_quantile = high_edge_quantile
        self.num_iter = num_iter
        self.min_roundness = min_roundness
        self.cluster_penalty = cluster_penalty
        self.roi_length = roi_length
        self.progress_bar = progress_bar
        self.search_timesteps = sorted(utils.to_list(search_timestep)) if search_timestep else [0]
        self.search_channels = utils.to_list(search_channel)

    def __call__(self, assay: xr.Dataset) -> xr.Dataset:
        if not self.search_channels:
            self.search_channels = assay.channel

        num_rows, num_cols = assay.tag.shape

        # Store each channel and timesteps for each marker in one chunk and set marker row/col
        # sizes so each chunk ends up being at least 1MB. We will rechunk later.
        chunk_bytes = 1e6
        # Don't take into account dtype size since fg/bg bool arrays should also be 1MB.
        roi_bytes = self.roi_length**2
        # Prioritize larger row chunks since we're more likely to want whole columns than rows.
        row_chunk_size = min(math.ceil(chunk_bytes / roi_bytes), num_rows)
        col_chunk_size = math.ceil(chunk_bytes / (roi_bytes * row_chunk_size))
        # Create the array of subimage regions.
        roi = da.empty(
            (
                num_rows,
                num_cols,
                assay.sizes["channel"],
                assay.sizes["time"],
                self.roi_length,
                self.roi_length,
            ),
            dtype=assay.image.dtype,
            chunks=(
                row_chunk_size,
                col_chunk_size,
                1,
                1,
                self.roi_length,
                self.roi_length,
            ),
        )
        assay["roi"] = (("mark_row", "mark_col", "channel", "time", "roi_y", "roi_x"), roi)
        assay = assay.assign_coords(
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
            x=(
                ("mark_row", "mark_col", "time"),
                np.empty((num_rows, num_cols, assay.sizes["time"])),
            ),
            y=(
                ("mark_row", "mark_col", "time"),
                np.empty((num_rows, num_cols, assay.sizes["time"])),
            ),
        )

        # Run the button finding algorithm for each timestep specified in search_timesteps.
        for t in tqdm.tqdm(self.search_timesteps, disable=not self.progress_bar):
            # Preload all images for this timestep so we only read from disk once.
            images = assay.image.isel(time=t).compute()
            # Find button centers.
            assay.x[..., t], assay.y[..., t] = self.find_centers(
                images.sel(channel=self.search_channels), assay
            )

            # Compute the roi, foreground and background masks for all buttons.
            (
                assay.roi[:, :, :, t],
                assay.fg[:, :, :, t],
                assay.bg[:, :, :, t],
                assay.x[..., t],
                assay.y[..., t],
                assay.valid[..., t],
            ) = self.find_rois(images, t, assay)
            # Eagerly compute the roi values so the dask task graph doesn't get too large.
            # TODO: Look into caching here or storing.
            assay["roi"] = assay.roi.persist()
            assay["fg"] = assay.fg.persist()
            assay["bg"] = assay.bg.persist()

        # Now fill in the remaining timesteps where we aren't searching.
        for t, time in enumerate(tqdm.tqdm(assay.time, disable=not self.progress_bar)):
            if t in self.search_timesteps:
                # Skip this timestep since we've already processed it.
                continue

            if t < self.search_timesteps[0]:
                # Backfill timesteps that come before the first searched timestep.
                copy_t = self.search_timesteps[0]
            else:
                # Re-use the button locations of the timestep just before this one since it's either
                # a searched timestep or a timestep we just copied locations into.
                copy_t = t - 1

            # Preload all images for this timestep so we only read from disk once and
            # convert all relevant data to numpy arrays since iterating through xarrays is slow.
            images = assay.image.sel(time=time).to_numpy()
            x = assay.x[..., copy_t].to_numpy()
            y = assay.y[..., copy_t].to_numpy()
            roi = np.empty_like(assay.roi[:, :, :, t])
            # Update the roi since the location is copied but the roi images are timestep specific.
            for i in range(num_rows):
                for j in range(num_cols):
                    top, bottom, left, right = utils.bounding_box(
                        round(x[i, j]),
                        round(y[i, j]),
                        roi.shape[-1],
                        assay.sizes["im_x"],
                        assay.sizes["im_y"],
                    )
                    roi[i, j] = images[..., top:bottom, left:right]

            assay.roi[:, :, :, t] = roi
            assay.fg[:, :, :, t] = assay.fg[:, :, :, copy_t]
            assay.bg[:, :, :, t] = assay.bg[:, :, :, copy_t]
            assay.x[..., t] = x
            assay.y[..., t] = y
            assay.valid[..., t] = assay.valid[..., copy_t]
            # Eagerly compute the roi values so the dask task graph doesn't get too large.
            # TODO: Look into caching here or storing.
            assay["roi"] = assay.roi.persist()
            assay["fg"] = assay.fg.persist()
            assay["bg"] = assay.bg.persist()
        assay = assay.stack(mark=("mark_row", "mark_col"), create_index=True).transpose("mark", ...)
        # Rechunk the array to chunk along markers since users will usually want to slice along that dimension.
        mark_chunk_size = min(
            math.ceil(chunk_bytes / (roi_bytes * assay.sizes["time"] * assay.sizes["channel"])),
            num_rows,
        )
        chunk_sizes = [
            mark_chunk_size,
            assay.sizes["channel"],
            assay.sizes["time"],
            assay.sizes["roi_y"],
            assay.sizes["roi_x"],
        ]
        # Cache the rechunked array to prevent delays.
        assay["roi"] = assay.roi.chunk(chunk_sizes).mg.cache()
        assay["fg"] = assay.fg.chunk(chunk_sizes).mg.cache()
        assay["bg"] = assay.bg.chunk(chunk_sizes).mg.cache()

        return assay

    def find_centers(self, images: xr.DataArray, assay: xr.Dataset):
        points = np.empty((0, 2))
        min_button_dist = round(min(self.row_dist, self.col_dist) / 2 - self.max_button_radius)
        if min_button_dist % 2 == 0:
            # Certain opencv functions require an odd blocksize.
            min_button_dist -= 1
        for image in images:
            image = utils.to_uint8(image.to_numpy())
            # Step 1: Find an imperfect button mask by thresholding.
            new_points = find_circles(
                image,
                low_edge_quantile=self.low_edge_quantile,
                high_edge_quantile=self.high_edge_quantile,
                grid_length=20,
                num_iter=self.num_iter,
                min_radius=self.min_button_radius,
                max_radius=self.max_button_radius,
                min_dist=min_button_dist,
                min_roundness=self.min_roundness,
            )[0][:, :2]

            if len(points) > 0:
                # Remove points too close to other points in previous channels.
                dist_matrix = np.linalg.norm(points[np.newaxis] - new_points[:, np.newaxis], axis=2)
                new_points = new_points[np.min(dist_matrix, axis=1) > min_button_dist]

            # Add the new points to the list of seen points.
            points = np.concatenate([points, new_points])

        # Split the points into x and y components.
        x = points[:, 1]
        y = points[:, 0]

        # Step 3: Cluster the points into distinct rows and columns.
        points_per_row = (assay.tag != "").sum(dim="mark_col").to_numpy()
        points_per_col = (assay.tag != "").sum(dim="mark_row").to_numpy()
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

    def find_rois(self, images: xr.DataArray, t: int, assay: xr.Dataset):
        # Convert all relevant quantities to numpy arrays since xarrays are very slow
        # when iterated over.
        images = images.to_numpy()
        num_rows, num_cols = assay.roi.shape[:2]
        tag = assay.tag.to_numpy()
        x = assay.x[:, :, t].to_numpy()
        y = assay.y[:, :, t].to_numpy()
        valid = assay.valid[:, :, t].to_numpy()
        roi = np.empty_like(assay.roi.isel(time=t))
        fg = np.empty_like(roi, dtype=bool)
        bg = np.empty_like(fg)
        search_channel_idxs = [
            list(assay.channel.to_numpy()).index(c) for c in self.search_channels
        ]

        for i in range(num_rows):
            for j in range(num_cols):
                # Initialize the roi image.
                top, bottom, left, right = utils.bounding_box(
                    round(x[i, j]),
                    round(y[i, j]),
                    roi.shape[-1],
                    assay.sizes["im_x"],
                    assay.sizes["im_y"],
                )
                roi[i, j] = images[..., top:bottom, left:right]

                best_circle = None
                best_score = -np.inf
                # Refine our button estimate unless we have a blank chamber.
                if tag[i, j] != "":
                    for channel in search_channel_idxs:
                        subimage = roi[i, j, channel]
                        subimage = utils.to_uint8(np.clip(subimage, np.median(subimage), None))
                        subimage -= subimage.min()
                        find_circles
                        circles, scores = find_circles(
                            subimage,
                            low_edge_quantile=self.low_edge_quantile,
                            high_edge_quantile=(
                                1 - np.pi * self.min_button_radius / self.roi_length**2
                            ),
                            grid_length=20,
                            num_iter=self.num_iter // (num_rows * num_cols),
                            min_radius=self.min_button_radius,
                            max_radius=self.max_button_radius,
                            min_dist=0,
                            min_roundness=self.min_roundness,
                        )
                        if len(circles) > 0:
                            scores = scores
                            idx = np.argmax(scores)
                            if scores[idx] > best_score:
                                best_circle = circles[idx]
                                best_score = scores[idx]

                # Update our estimate of the button position if we found some circles.
                if best_circle is not None:
                    y[i, j], x[i, j] = best_circle[:2]
                    # Change coordinates from roi to image coordinates.
                    x[i, j] += left
                    y[i, j] += top
                    # Move the roi bounding box to center the new x, y values.
                    top, bottom, left, right = utils.bounding_box(
                        round(x[i, j]),
                        round(y[i, j]),
                        self.roi_length,
                        assay.sizes["im_x"],
                        assay.sizes["im_y"],
                    )
                    roi[i, j] = images[..., top:bottom, left:right]

                x_rel = round(x[i, j]) - left
                y_rel = round(y[i, j]) - top

                # Set the background to be the annulus around a circle of fixed radius.
                bg_mask = utils.annulus(
                    self.roi_length,
                    row=y_rel,
                    col=x_rel,
                    outer_radius=2 * self.max_button_radius,
                    inner_radius=self.max_button_radius,
                    value=1,
                )

                fg_mask = utils.circle(
                    self.roi_length,
                    row=y_rel,
                    col=x_rel,
                    radius=self.max_button_radius,
                    value=1,
                )

                fg[i, j] = fg_mask
                bg[i, j] = bg_mask

        return roi, fg, bg, x, y, valid

    @registry.components.register("find_buttons")
    def make(
        row_dist: float = 375 / 3.22,
        col_dist: float = 655 / 3.22,
        min_button_radius: int = 4,
        max_button_radius: int = 15,
        low_edge_quantile: float = 0.1,
        high_edge_quantile: float = 0.9,
        num_iter: int = 5000000,
        min_roundness: float = 0.2,
        cluster_penalty: float = 50,
        roi_length: int = 61,
        progress_bar: bool = False,
        search_timestep: list[int] | None = None,
        search_channel: str | list[str] | None = None,
    ):
        return ButtonFinder(
            row_dist=row_dist,
            col_dist=col_dist,
            min_button_radius=min_button_radius,
            max_button_radius=max_button_radius,
            low_edge_quantile=low_edge_quantile,
            high_edge_quantile=high_edge_quantile,
            num_iter=num_iter,
            min_roundness=min_roundness,
            cluster_penalty=cluster_penalty,
            roi_length=roi_length,
            progress_bar=progress_bar,
            search_timestep=search_timestep,
            search_channel=search_channel,
        )


class BeadFinder:
    def __init__(
        self,
        min_bead_radius: int = 5,
        max_bead_radius: int = 25,
        low_edge_quantile: float = 0.1,
        high_edge_quantile: float = 0.9,
        num_iter: int = 5000000,
        min_roundness: float = 0.3,
        roi_length: int = 61,
        search_channel: str | list[str] | None = None,
    ):
        self.min_bead_radius = min_bead_radius
        self.max_bead_radius = max_bead_radius
        self.low_edge_quantile = low_edge_quantile
        self.high_edge_quantile = high_edge_quantile
        self.num_iter = num_iter
        self.min_roundness = min_roundness
        self.roi_length = roi_length
        self.search_channels = utils.to_list(search_channel)

    def __call__(self, assay: xr.Dataset) -> xr.Dataset:
        if not self.search_channels:
            self.search_channels = assay.channel

        beads = np.empty((0, 3))
        for t in assay.time:
            for search_channel in self.search_channels:
                image = utils.to_uint8(assay.image.sel(channel=search_channel, time=t).to_numpy())
                b = find_circles(
                    image,
                    low_edge_quantile=self.low_edge_quantile,
                    high_edge_quantile=self.high_edge_quantile,
                    grid_length=20,
                    num_iter=self.num_iter,
                    min_radius=self.min_bead_radius,
                    max_radius=self.max_bead_radius,
                    min_dist=2 * self.min_bead_radius,
                    min_roundness=self.min_roundness,
                )[0]
                if len(beads) > 0:
                    # Exclude beads that we've already seen.
                    duplicates = np.array(
                        [
                            len(neighbors) > 0
                            for neighbors in scipy.spatial.KDTree(beads[:, :2]).query_ball_point(
                                b[:, :2], 2 * self.min_bead_radius
                            )
                        ]
                    )
                    b = b[~duplicates]
                beads = np.concatenate([beads, b])

        num_beads = len(beads)
        # Store each channel and timesteps for each marker in one chunk and set marker row/col
        # sizes so each chunk ends up being at least 1MB.
        chunk_bytes = 1e6
        # Don't take into account dtype size since fg/bg bool arrays should also be 1MB.
        roi_bytes = self.roi_length**2
        # Create the array of subimage regions.
        roi = da.empty(
            (
                num_beads,
                assay.sizes["channel"],
                assay.sizes["time"],
                self.roi_length,
                self.roi_length,
            ),
            dtype=assay.image.dtype,
            chunks=(
                min(
                    math.ceil(
                        chunk_bytes / (roi_bytes * assay.sizes["channel"] * assay.sizes["time"])
                    ),
                    num_beads,
                ),
                assay.sizes["channel"],
                assay.sizes["time"],
                self.roi_length,
                self.roi_length,
            ),
        )

        assay["roi"] = (("mark", "channel", "time", "roi_y", "roi_x"), roi)
        assay = assay.assign_coords(
            fg=(
                ("mark", "channel", "time", "roi_y", "roi_x"),
                da.empty_like(roi, dtype=bool),
            ),
            bg=(
                ("mark", "channel", "time", "roi_y", "roi_x"),
                da.empty_like(roi, dtype=bool),
            ),
            x=(
                ("mark", "time"),
                np.repeat(beads[:, np.newaxis, 1], assay.sizes["time"], axis=1),
            ),
            y=(
                ("mark", "time"),
                np.repeat(beads[:, np.newaxis, 0], assay.sizes["time"], axis=1),
            ),
        )

        # Create a label array that contains the areas owned by each bead.
        labels = circle_labels(beads.astype(int), assay.sizes["im_y"], assay.sizes["im_x"])

        # Compute the foreground and background masks for all buttons.
        # TODO: Don't assume beads don't move across timesteps.
        # Iterate over numpy arrays since indexing over xarrays is slow.
        x = assay.x.sel(time=0).to_numpy()
        y = assay.y.sel(time=0).to_numpy()
        fg = np.empty((num_beads,) + assay.fg.shape[2:], dtype=bool)
        bg = np.empty_like(fg)
        image = assay.image.sel(time=0, channel=self.search_channels).to_numpy()
        for i in range(num_beads):
            # Set the subimage region for this bead.
            top, bottom, left, right = utils.bounding_box(
                round(x[i]),
                round(y[i]),
                self.roi_length,
                assay.sizes["im_x"],
                assay.sizes["im_y"],
            )
            sublabels = labels[top:bottom, left:right]
            # Set the foreground of the bead.
            fg[i] = sublabels == i
            # Set the background to be the region assigned to no beads.
            bg[i] = sublabels == -1
        assay.fg[:] = fg[:, np.newaxis]
        assay.bg[:] = bg[:, np.newaxis]

        # Individually add each channel to save on memory.
        for i, channel in enumerate(assay.channel):
            image = assay.image.sel(channel=channel).to_numpy()
            roi = np.empty((num_beads,) + assay.roi.shape[2:], dtype=assay.roi.dtype)
            for j in range(num_beads):
                # Set the subimage region for this bead.
                top, bottom, left, right = utils.bounding_box(
                    round(x[j]),
                    round(y[j]),
                    self.roi_length,
                    assay.sizes["im_x"],
                    assay.sizes["im_y"],
                )
                roi[j] = image[..., top:bottom, left:right]
            assay.roi[:, i] = roi

        assay["roi"] = assay.roi.mg.cache()
        assay["fg"] = assay.fg.mg.cache()
        assay["bg"] = assay.bg.mg.cache()
        assay = assay.assign_coords(
            valid=(
                ("mark", "time"),
                np.ones((assay.sizes["mark"], assay.sizes["time"]), dtype=bool),
            ),
        )
        return assay

    @registry.components.register("find_beads")
    def make(
        min_bead_radius: int = 5,
        max_bead_radius: int = 25,
        low_edge_quantile: float = 0.1,
        high_edge_quantile: float = 0.9,
        num_iter: int = 5000000,
        min_roundness: float = 0.3,
        roi_length: int = 61,
        search_channel: str | list[str] | None = None,
    ):
        return BeadFinder(
            min_bead_radius=min_bead_radius,
            max_bead_radius=max_bead_radius,
            low_edge_quantile=low_edge_quantile,
            high_edge_quantile=high_edge_quantile,
            num_iter=num_iter,
            min_roundness=min_roundness,
            roi_length=roi_length,
            search_channel=search_channel,
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

    min_cost = np.inf
    best_spans = None
    for offset in range(total_length - round(num_clusters * cluster_length)):
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
        if cost.sum() < min_cost:
            min_cost = cost.sum()
            best_spans = spans

    # Label each point with its cluster, label points outside clusters as -1.
    labels = -np.ones_like(points, dtype=int)
    labels[best_spans[0] : best_spans[-1]] = np.repeat(
        np.arange(num_clusters), best_spans[1:] - best_spans[:-1]
    )

    # Return the labels based on the original order of the points.
    return labels[np.argsort(permutation)]


def regress_clusters(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
    ideal_num_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if num_clusters == 1:
        # If we only have a single cluster then we can't average regression results.
        if len(x) == 1:
            return 0, y
        else:
            return scipy.stats.linregress(x, y)[:2]

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
):
    # TODO: Make this functions nicer.
    # Step 1: Denoise the image for more accurate edge finding.
    img = cv.GaussianBlur(img, (5, 5), 0)
    logger.debug("Blurred Image", img)

    # Step 2: Find edges from image gradients.
    dx = cv.Scharr(img, ddepth=cv.CV_32F, dx=1, dy=0)
    dy = cv.Scharr(img, ddepth=cv.CV_32F, dx=0, dy=1)
    grad = np.sqrt(dx**2 + dy**2)
    logger.debug("Gradient Magnitudes", grad)
    low_thresh = np.quantile(grad, low_edge_quantile)
    high_thresh = np.quantile(grad, high_edge_quantile)
    edges = cv.Canny(
        dx.astype(np.int16),
        dy.astype(np.int16),
        threshold1=low_thresh,
        threshold2=high_thresh,
        L2gradient=True,
    )
    edges[edges != 0] = 1
    logger.debug(f"Edges (low_thresh: {low_thresh} high_thresh: {high_thresh})", edges)

    # Step 3: Use edges to find candidate circles.
    circles = candidate_circles(edges, grid_length, num_iter)

    # Step 4: Filter circles based on size and position.
    # Remove circles that are too small or large.
    circles = circles[(circles[:, 2] >= min_radius) & (circles[:, 2] <= max_radius)]
    # Round circle coordinates since we'll only by considering whole pixels.
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
    edges = np.pad(edges, pad)
    # Adjust circle coordinates to account for padding.
    circles[:, :2] += pad
    # Sort circles by radius.
    order = np.argsort(circles[:, 2])
    circles = circles[order]

    start = 0
    scores = []
    for radius in range(min_radius, max_radius + 1):
        perimeter_coords = utils.circle_points(radius)
        end = np.searchsorted(circles[:, 2], radius + 1)
        s = mean_grad(thetas, edges, circles[start:end, :2], perimeter_coords)
        scores.append(s / len(perimeter_coords))
        start = end
    circles[:, :2] -= pad
    scores = np.concatenate(scores)
    circles = circles[scores >= min_roundness]
    scores = scores[scores >= min_roundness]

    # Step 6: Remove duplicate circles that are too close to each other.
    perm = np.argsort(-scores)
    if min_dist > 0:
        circles = filter_neighbors(circles[perm], min_dist)

    return circles, scores


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
    coords = utils.circle_points(min_dist, four_connected=True)

    pad = 2 * min_dist + 1
    arr = -np.ones((circles[:, 0].max() + 2 * pad, circles[:, 1].max() + 2 * pad), dtype=np.int32)
    valid = np.ones(len(circles), dtype=np.bool_)
    for i in range(len(circles)):
        for j in prange(len(coords)):
            row = coords[j, 0] + circles[i, 0] + pad
            col = coords[j, 1] + circles[i, 1] + pad
            v = arr[row, col]
            if v != -1:
                valid[i] = False
                break
        if not valid[i]:
            continue
        for j in prange(len(coords)):
            row = coords[j, 0] + circles[i, 0] + pad
            col = coords[j, 1] + circles[i, 1] + pad
            arr[row, col] = i

    return circles[valid]


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
                i * grid_length : (i + 1) * grid_length, j * grid_length : (j + 1) * grid_length
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
                    i * grid_length : (i + 1) * grid_length, j * grid_length : (j + 1) * grid_length
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
        pts = utils.filled_circle_points(circles[i, 2])
        pts += circles[i, :2]
        for j in prange(len(pts)):
            r, c = pts[j]
            if r >= 0 and r < num_rows and c >= 0 and c < num_cols:
                if labels[r, c] != -1:
                    labels[r, c] = -2
                else:
                    labels[r, c] = i

    return labels
