from __future__ import annotations

import math
import logging

from numpy.typing import ArrayLike
import cv2 as cv
import dask.array as da
import numba
import numpy as np
import scipy
import sklearn.mixture
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
        min_contrast: int | None = None,
        min_roundness: float = 0.75,
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
        self.min_roundness = min_roundness
        self.cluster_penalty = cluster_penalty
        self.roi_length = roi_length
        self.progress_bar = progress_bar
        self.search_timesteps = sorted(utils.to_list(search_timestep)) if search_timestep else [0]
        self.search_channels = utils.to_list(search_channel)
        self.min_contrast = min_contrast

    def __call__(self, assay: xr.Dataset) -> xr.Dataset:
        if not self.search_channels:
            self.search_channels = assay.channel

        num_rows, num_cols = assay.tag.shape

        # Store each channel and timesteps for each marker in one chunk and set marker row/col
        # sizes so each chunk ends up being at least 50MB. We will rechunk later.
        chunk_bytes = 5e7
        # Don't take into account dtype size since fg/bg bool arrays should also be 50MB.
        roi_bytes = self.roi_length**2
        # Prioritize larger row chunks since we're more likely to want whole columns than rows.
        row_chunk_size = min(math.ceil(chunk_bytes / roi_bytes), num_rows)
        col_chunk_size = math.ceil(chunk_bytes / (roi_bytes * row_chunk_size))
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
                np.empty((num_rows, num_cols, assay.dims["time"])),
            ),
            y=(
                ("mark_row", "mark_col", "time"),
                np.empty((num_rows, num_cols, assay.dims["time"])),
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
            assay["roi"] = assay.roi.persist()
            assay["fg"] = assay.fg.persist()
            assay["bg"] = assay.bg.persist()
        assay = assay.stack(mark=("mark_row", "mark_col"), create_index=True).transpose("mark", ...)
        # Rechunk the array to chunk along markers since users will usually want to slice along that dimension.
        mark_chunk_size = min(
            math.ceil(chunk_bytes / (roi_bytes * assay.dims["time"] * assay.dims["channel"])),
            num_rows,
        )
        chunk_sizes = [
            mark_chunk_size,
            assay.dims["channel"],
            assay.dims["time"],
            assay.dims["roi_y"],
            assay.dims["roi_x"],
        ]
        # Eagerly compute the rechunking to prevent delays.
        assay["roi"] = assay.roi.chunk(chunk_sizes).persist()
        assay["fg"] = assay.fg.chunk(chunk_sizes).persist()
        assay["bg"] = assay.bg.chunk(chunk_sizes).persist()

        return assay

    def find_centers(self, images: xr.DataArray, assay: xr.Dataset):
        points = np.empty((0, 2))
        min_button_dist = round(min(self.row_dist, self.col_dist) / 2)
        if min_button_dist % 2 == 0:
            # Certain opencv functions require an odd blocksize.
            min_button_dist -= 1
        for image in images:
            image_max = image.max().item()
            image = utils.to_uint8(image.to_numpy())
            # Step 1: Find an imperfect button mask by thresholding.
            blur = cv.medianBlur(image, 2 * min_button_dist + 1)
            if self.min_contrast is None:
                # Guess a reasonable threshold for buttons.
                num_points = (assay.tag != "").sum()
                q = np.pi * self.max_button_radius**2 * num_points / image.size
                thresh = np.quantile(np.maximum(image.astype(np.int16) - blur, 0), 1 - 2 * q)
            else:
                thresh = 255 * self.min_contrast / image_max
            mask = utils.to_uint8(image > blur + thresh)

            # Step 2: Find connected components and filter out points.
            _, _, stats, new_points = cv.connectedComponentsWithStats(mask, connectivity=4)

            # Ignore the background point.
            new_points = new_points[1:]
            stats = stats[1:]

            # Exclude large and small blobs.
            new_points = new_points[
                (stats[:, cv.CC_STAT_HEIGHT] <= 2 * self.max_button_radius)
                & (stats[:, cv.CC_STAT_WIDTH] <= 2 * self.max_button_radius)
                & (stats[:, cv.CC_STAT_AREA] <= np.pi * self.max_button_radius**2)
                & (stats[:, cv.CC_STAT_HEIGHT] >= 2 * self.min_button_radius)
                & (stats[:, cv.CC_STAT_WIDTH] >= 2 * self.min_button_radius)
                & (stats[:, cv.CC_STAT_AREA] >= np.pi * self.min_button_radius**2)
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
        offsets = np.zeros((num_rows, num_cols, 2), dtype=int)

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

                best_subimage = None
                best_contour = None
                best_roundness = 0
                # Refine our button estimate unless we have a blank chamber.
                if tag[i, j] != "":
                    for channel in search_channel_idxs:
                        subimage = roi[i, j, channel]
                        subimage = utils.to_uint8(np.clip(subimage, np.median(subimage), None))
                        subimage -= subimage.min()
                        if self.min_contrast is None:
                            _, mask = cv.threshold(
                                subimage, thresh=0, maxval=1, type=cv.THRESH_BINARY + cv.THRESH_OTSU
                            )
                        else:
                            thresh = 255 * self.min_contrast / roi[i, j, channel].max()
                            mask = utils.to_uint8(subimage > thresh)

                        contours, hierarchy = cv.findContours(
                            mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE
                        )
                        hierarchy = hierarchy[0]
                        for c, h in zip(contours, hierarchy):
                            # Don't consider holes.
                            if h[3] == 0:
                                continue
                            perimeter = cv.arcLength(c, True)
                            area = cv.contourArea(c)
                            if h[2] != -1:
                                area -= cv.contourArea(contours[h[2]])
                                # Save the contour of the hole too.
                                c = [c, contours[h[2]]]
                            else:
                                # Save contour as element of a list for consistency.
                                c = [c]
                            # Don't consider contours that are the wrong size.
                            if (
                                perimeter <= 2 * np.pi * self.min_button_radius
                                or perimeter >= 2 * np.pi * self.max_button_radius
                                or area <= np.pi * self.min_button_radius**2
                                or area >= np.pi * self.max_button_radius**2
                            ):
                                continue
                            # Save this contour if it's the roundest one.
                            roundness = 4 * np.pi * area / perimeter**2
                            if i == 3 and j == 25:
                            if roundness > best_roundness and roundness > self.min_roundness:
                                best_roundness = roundness
                                best_contour = c
                                best_subimage = subimage

                # Update our estimate of the button position if we found some circles.
                if best_contour is not None:
                    y[i, j], x[i, j] = utils.contour_center(best_contour[0])
                    # Refine the background by removing bright areas.
                    # Change coordinates from roi to image coordinates.
                    x[i, j] += left
                    y[i, j] += top
                    # Move the roi bounding box to center the new x, y values.
                    old_top = top
                    old_left = left
                    top, bottom, left, right = utils.bounding_box(
                        round(x[i, j]),
                        round(y[i, j]),
                        self.roi_length,
                        assay.sizes["im_x"],
                        assay.sizes["im_y"],
                    )
                    roi[i, j] = images[..., top:bottom, left:right]
                    # Move contour coordinates to new centers.
                    for c in best_contour:
                        c[:, :, 0] += old_left - left
                        c[:, :, 1] += old_top - top

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

                if best_contour is not None:
                    fg_mask = np.zeros(fg.shape[-2:], dtype=np.uint8)
                    cv.drawContours(fg_mask, best_contour, -1, 1, cv.FILLED)
                    # Refine the background by removing bright areas.
                    _, bright_mask = cv.threshold(
                        best_subimage, thresh=0, maxval=1, type=cv.THRESH_BINARY + cv.THRESH_OTSU
                    )
                    dim_mask = 1 - cv.dilate(
                        bright_mask, np.ones((self.max_button_radius, self.max_button_radius))
                    )
                    if np.any(bg_mask * dim_mask):
                        bg_mask *= dim_mask
                    valid[i, j] = True
                else:
                    # If we didn't find a suitable foreground just set it to be a large circle.
                    fg_mask = utils.circle(
                        self.roi_length,
                        row=y_rel,
                        col=x_rel,
                        radius=self.max_button_radius,
                        value=1,
                    )
                    valid[i, j] = False

                fg[i, j] = fg_mask
                bg[i, j] = bg_mask

        return roi, fg, bg, x, y, valid

    @registry.components.register("find_buttons")
    def make(
        row_dist: float = 375 / 3.22,
        col_dist: float = 655 / 3.22,
        min_button_radius: int = 4,
        max_button_radius: int = 15,
        min_contrast: int | None = None,
        min_roundness: float = 0.75,
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
            min_contrast=min_contrast,
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
        roi_length: int = 61,
        search_channel: str | list[str] | None = None,
    ):
        self.min_bead_radius = min_bead_radius
        self.max_bead_radius = max_bead_radius
        self.roi_length = roi_length
        self.search_channels = utils.to_list(search_channel)

    def __call__(self, assay: xr.Dataset) -> xr.Dataset:
        if not self.search_channels:
            self.search_channels = assay.channel

        centers = np.empty((0, 2))
        labels = np.zeros((assay.sizes["im_y"], assay.sizes["im_x"]), dtype=int)
        for t in assay.time:
            for search_channel in self.search_channels:
                image = utils.to_uint8(assay.image.sel(channel=search_channel, time=t).to_numpy())
                # Step 1: Denoise the image for more accurate edge finding.
                image = cv.GaussianBlur(image, (5, 5), 0)

                # Step 2: Find edges from image gradients.
                dx = cv.Scharr(image, ddepth=cv.CV_16S, dx=1, dy=0)
                dy = cv.Scharr(image, ddepth=cv.CV_16S, dx=0, dy=1)
                grad = np.abs(dx) + np.abs(dy)
                # Assuming at least 1% of the image contains beads the local maxima will usually
                # be an edge so use that to set a reasonable upper threshold for edge finding.
                grad_max = cv.dilate(
                    grad, np.ones((10 * self.max_bead_radius, 10 * self.max_bead_radius))
                )
                # The median gradient should be background so use it as the lower threshold.
                edges = 255 - cv.Canny(
                    dx=dx, dy=dy, threshold1=np.median(grad), threshold2=np.median(grad_max) / 3
                )

                # Step 3: Treat edges as boundaries to find connected components that are either
                # all background or foreground.
                edge_width = 2 * self.min_bead_radius
                kernel = cv.getStructuringElement(
                    cv.MORPH_ELLIPSE, (edge_width + 1, edge_width + 1)
                )
                # Thicken edges to make sure they will be closed.
                edges = cv.erode(edges, kernel)
                _, l, stats, c = cv.connectedComponentsWithStats(
                    edges, connectivity=8, ltype=cv.CV_16U
                )

                # Step 4: Filter out regions that don't meet size criteria or aren't bright enough.
                min_radius = self.min_bead_radius - edge_width
                # Take into account the fact that beads are smaller because we grew edges.
                is_fg = (
                    (stats[:, cv.CC_STAT_WIDTH] >= min_radius)
                    & (stats[:, cv.CC_STAT_HEIGHT] >= min_radius)
                    & (stats[:, cv.CC_STAT_AREA] >= 2 * np.pi * min_radius)
                    & (stats[:, cv.CC_STAT_WIDTH] <= 2 * self.max_bead_radius + 200)
                    & (stats[:, cv.CC_STAT_HEIGHT] <= 2 * self.max_bead_radius)
                    & (stats[:, cv.CC_STAT_AREA] <= np.pi * self.max_bead_radius**2)
                )

                # Estimate the median background and foreground intensity.
                fg_vals = []
                bg_vals = []
                for i in range(1, l.max() + 1):
                    if is_fg[i]:
                        fg_vals.append(np.median(image[l == i]))
                    else:
                        bg_vals.append(image[l == i])
                fg_vals = np.array(fg_vals)
                bg_vals = np.concatenate(bg_vals)
                fg_median = np.median(fg_vals)
                bg_median = np.median(bg_vals)

                # Exclude foreground components whose brightness is similar to background if the
                # distribution of foregrounds looks bimodal.
                vals = np.concatenate([fg_vals, bg_vals[: len(fg_vals)]])[:, np.newaxis]
                fg_vals = fg_vals[:, np.newaxis]
                is_fg[is_fg] = (
                    sklearn.mixture.GaussianMixture(
                        n_components=2, means_init=([[bg_median], [fg_median]])
                    )
                    .fit(vals)
                    .predict(fg_vals)
                    == 1
                )

                # Reindex labels to be contiguous and set all non-foregrounds to 0.
                label_idx = 1
                for i in range(1, l.max() + 1):
                    if is_fg[i]:
                        l[l == i] = label_idx
                        label_idx += 1
                    else:
                        # TODO: We should handle edges differently since they're not backgrounds.
                        l[l == i] = 0

                # Redilate all foreground components to account for the erosion we did earlier.
                l = cv.dilate(l, kernel)

                # Step 5: Add new beads to previous channels' results.
                c = c[is_fg]
                if len(centers) > 0:
                    # Exclude beads that we've already seen.
                    duplicates = np.array(
                        [
                            len(neighbors) > 0
                            for neighbors in scipy.spatial.KDTree(centers).query_ball_point(
                                c, 2 * self.min_bead_radius
                            )
                        ]
                    )
                    c = c[~duplicates]
                    l[np.isin(l, np.where(duplicates)[0] + 1)] = 0
                    l = np.unique(l, return_inverse=True)[1].reshape(l.shape)
                centers = np.concatenate([centers, c])
                nnz = l != 0
                labels[nnz] = l[nnz] + labels.max()

        num_beads = len(centers)
        # Store each channel and timesteps for each marker in one chunk and set marker row/col
        # sizes so each chunk ends up being at least 50MB. We will rechunk later.
        chunk_bytes = 5e7
        # Don't take into account dtype size since fg/bg bool arrays should also be 50MB.
        roi_bytes = self.roi_length**2
        # Create the array of subimage regions.
        roi = da.empty(
            (
                num_beads,
                assay.dims["channel"],
                assay.dims["time"],
                self.roi_length,
                self.roi_length,
            ),
            dtype=assay.image.dtype,
            chunks=(
                min(
                    math.ceil(
                        chunk_bytes / (roi_bytes * assay.dims["channel"] * assay.dims["time"])
                    ),
                    num_beads,
                ),
                assay.dims["channel"],
                assay.dims["time"],
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
                np.repeat(centers[:, np.newaxis, 0], assay.dims["time"], axis=1),
            ),
            y=(
                ("mark", "time"),
                np.repeat(centers[:, np.newaxis, 1], assay.dims["time"], axis=1),
            ),
            valid=(
                ("mark", "time"),
                np.ones((assay.sizes["mark"], assay.sizes["time"]), dtype=bool),
            ),
        )

        # Compute the foreground and background masks for all buttons.
        # TODO: Don't assume beads don't move across timesteps.
        # Iterate over numpy arrays since indexing over xarrays is slow.
        x = assay.x.sel(time=0).to_numpy()
        y = assay.y.sel(time=0).to_numpy()
        fg = np.empty((num_beads,) + assay.fg.shape[2:], dtype=bool)
        bg = np.empty_like(fg)
        for i in range(num_beads):
            # Set the subimage region for this bead.
            top, bottom, left, right = utils.bounding_box(
                round(x[i]),
                round(y[i]),
                self.roi_length,
                assay.sizes["im_x"],
                assay.sizes["im_y"],
            )
            # Set the foreground of the bead.
            fg[i] = labels[top:bottom, left:right] == i + 1
            # Set the background to be the region assigned to no beads.
            bg[i] = labels[top:bottom, left:right] == 0
        assay.fg[:] = fg[:, np.newaxis]
        assay.bg[:] = bg[:, np.newaxis]

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

        assay["roi"] = assay.roi.persist()
        assay["fg"] = assay.fg.persist()
        assay["bg"] = assay.bg.persist()
        assay = assay.assign_coords()
        return assay

    @registry.components.register("find_beads")
    def make(
        min_bead_radius: int = 5,
        max_bead_radius: int = 25,
        roi_length: int = 61,
        search_channel: str | list[str] | None = None,
    ):
        return BeadFinder(
            min_bead_radius=min_bead_radius,
            max_bead_radius=max_bead_radius,
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


@numba.jit(nopython=True)
def get_label_centers(labels):
    num_labels = labels.max() + 1
    centers = np.zeros((num_labels, 2))
    num_points = np.zeros(num_labels)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            k = labels[i, j]
            centers[k, 0] += j
            centers[k, 1] += i
            num_points[k] += 1
    centers /= np.expand_dims(num_points, axis=1)
    return centers[1:]
