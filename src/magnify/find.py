import math

import dask.array as da
import numpy as np
import scipy
import tqdm
import xarray as xr

from magnify import registry, utils
from magnify.plot.vis import InteractiveUI


class ButtonFinder:
    def __init__(
        self,
        row_dist: float,
        col_dist: float,
        min_button_diameter: int,
        max_button_diameter: int,
        chamber_diameter: int,
        top_chamber: int | None,
        left_chamber: int | None,
        low_edge_quantile: float,
        high_edge_quantile: float,
        num_iter: int,
        min_roundness: float,
        cluster_penalty: float,
        roi_length: int | None,
        progress_bar: bool,
        search_timestep: int | list[int],
        search_channel: str | list[str] | None,
        interactive: bool,
    ):
        self.row_dist = row_dist
        self.col_dist = col_dist
        self.min_button_radius = math.floor(min_button_diameter / 2)
        self.max_button_radius = math.ceil(max_button_diameter / 2)
        self.chamber_radius = round(chamber_diameter / 2)
        self.top_chamber = top_chamber
        self.left_chamber = left_chamber
        self.low_edge_quantile = low_edge_quantile
        self.high_edge_quantile = high_edge_quantile
        self.num_iter = num_iter
        self.min_roundness = min_roundness
        self.cluster_penalty = cluster_penalty
        self.roi_length = roi_length if roi_length is not None else round(1.2 * chamber_diameter)
        self.progress_bar = progress_bar
        self.gui = InteractiveUI() if interactive else None
        self.search_timesteps = sorted(utils.to_list(search_timestep))
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
        assay["roi"] = (
            ("mark_row", "mark_col", "channel", "time", "roi_y", "roi_x"),
            roi,
        )
        assay = assay.assign_coords(
            fg=(
                ("mark_row", "mark_col", "time", "roi_y", "roi_x"),
                da.empty_like(
                    roi,
                    dtype=bool,
                )[:, :, 0],
            ),
            bg=(
                ("mark_row", "mark_col", "time", "roi_y", "roi_x"),
                da.empty_like(
                    roi,
                    dtype=bool,
                )[:, :, 0],
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
                assay.fg[:, :, t],
                assay.bg[:, :, t],
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
            assay.fg[:, :, t] = assay.fg[:, :, copy_t]
            assay.bg[:, :, t] = assay.bg[:, :, copy_t]
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
        chunk_sizes = {
            "mark": mark_chunk_size,
            "channel": assay.sizes["channel"],
            "time": assay.sizes["time"],
            "roi_y": assay.sizes["roi_y"],
            "roi_x": assay.sizes["roi_x"],
        }
        # Cache the rechunked array to prevent delays.
        assay["roi"] = assay.roi.chunk(chunk_sizes)
        chunk_sizes.pop("channel")
        assay["fg"] = assay.fg.chunk(chunk_sizes)
        assay["bg"] = assay.bg.chunk(chunk_sizes)
        assay.mg.cache(["roi", "fg", "bg"])

        return assay

    def find_centers(self, images: xr.DataArray, assay: xr.Dataset):
        points = np.empty((0, 2))
        min_button_dist = self.chamber_radius
        for image in images:
            image = utils.to_uint8(image.to_numpy())
            # Step 1: Find an imperfect button mask through circle finding.
            new_points = utils.find_circles(
                image,
                low_edge_quantile=self.low_edge_quantile,
                high_edge_quantile=self.high_edge_quantile,
                grid_length=20,
                num_iter=self.num_iter,
                min_radius=self.min_button_radius,
                max_radius=self.max_button_radius,
                min_dist=min_button_dist,
                min_roundness=self.min_roundness,
                gui=self.gui,
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
        if self.top_chamber is None:
            row_labels = cluster_1d(
                y,
                total_length=image.shape[0],
                num_clusters=num_rows,
                cluster_length=self.row_dist,
                ideal_num_points=points_per_row,
                penalty=self.cluster_penalty,
            )
        else:
            # We have the top boundary of the chip so we can use that to find the optimal cluster.
            row_labels = label_clusters(
                y,
                offset=self.top_chamber,
                num_clusters=num_rows,
                cluster_length=2 * self.chamber_radius,
                cluster_gap=self.row_dist - 2 * self.chamber_radius,
            )

        if self.left_chamber is None:
            col_labels = cluster_1d(
                x,
                total_length=image.shape[1],
                num_clusters=num_cols,
                cluster_length=self.col_dist,
                ideal_num_points=points_per_col,
                penalty=self.cluster_penalty,
            )
        else:
            # We have the left boundary of the chip so we can use that to find the optimal cluster.
            col_labels = label_clusters(
                x,
                offset=self.left_chamber,
                num_clusters=num_cols,
                cluster_length=2 * self.chamber_radius,
                cluster_gap=self.col_dist - 2 * self.chamber_radius,
            )

        # Exclude boundary points that didn't fall into clusters.
        in_cluster = (row_labels >= 0) & (col_labels >= 0)
        x, y = x[in_cluster], y[in_cluster]
        col_labels, row_labels = col_labels[in_cluster], row_labels[in_cluster]

        # Step 4: Draw lines through each cluster.
        row_slope, row_intercepts = regress_clusters(
            x,
            y,
            labels=row_labels,
            num_clusters=num_rows,
            ideal_num_points=points_per_row,
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
        fg = np.empty_like(assay.fg.isel(time=t), dtype=bool)
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
                        subimage = utils.to_uint8(roi[i, j, channel])
                        circles, scores = utils.find_circles(
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
                            gui=None,
                        )
                        if len(circles) > 0:
                            scores = scores
                            idx = np.argmax(scores)
                            if scores[idx] > best_score:
                                best_circle = circles[idx]
                                best_score = scores[idx]

                # Update our estimate of the button position if we found some circles.
                button_radius = self.max_button_radius
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
                    button_radius = best_circle[2]

                x_rel = round(x[i, j]) - left
                y_rel = round(y[i, j]) - top

                # Set the background to be the annulus around a circle of fixed radius.
                bg_mask = utils.annulus(
                    (self.roi_length, self.roi_length),
                    (y_rel, x_rel),
                    outer_radius=self.chamber_radius,
                    inner_radius=self.max_button_radius,
                    value=1,
                )

                fg_mask = utils.circle(
                    (self.roi_length, self.roi_length),
                    (y_rel, x_rel),
                    radius=button_radius,
                    value=1,
                )

                fg[i, j] = fg_mask
                bg[i, j] = bg_mask

        return roi, fg, bg, x, y, valid

    @registry.components.register("find_buttons")
    def make(
        row_dist: float,
        col_dist: float,
        min_button_diameter: int,
        max_button_diameter: int,
        chamber_diameter: int,
        top_chamber: int | None,
        left_chamber: int | None,
        low_edge_quantile: float,
        high_edge_quantile: float,
        num_iter: int,
        min_roundness: float,
        cluster_penalty: float,
        roi_length: int | None,
        progress_bar: bool,
        search_timestep: int | list[int],
        search_channel: str | list[str] | None,
        interactive: bool,
    ):
        return ButtonFinder(
            row_dist=row_dist,
            col_dist=col_dist,
            min_button_diameter=min_button_diameter,
            max_button_diameter=max_button_diameter,
            chamber_diameter=chamber_diameter,
            top_chamber=top_chamber,
            left_chamber=left_chamber,
            low_edge_quantile=low_edge_quantile,
            high_edge_quantile=high_edge_quantile,
            num_iter=num_iter,
            min_roundness=min_roundness,
            cluster_penalty=cluster_penalty,
            roi_length=roi_length,
            progress_bar=progress_bar,
            search_timestep=search_timestep,
            search_channel=search_channel,
            interactive=interactive,
        )


class BeadFinder:
    def __init__(
        self,
        min_bead_diameter: int,
        max_bead_diameter: int,
        low_edge_quantile: float,
        high_edge_quantile: float,
        num_iter: int,
        min_roundness: float,
        roi_length: int | None,
        search_channel: str | list[str] | None,
        interactive: bool,
    ):
        self.min_bead_radius = math.floor(min_bead_diameter / 2)
        self.max_bead_radius = math.ceil(max_bead_diameter / 2)
        self.low_edge_quantile = low_edge_quantile
        self.high_edge_quantile = high_edge_quantile
        self.num_iter = num_iter
        self.min_roundness = min_roundness
        self.roi_length = roi_length if roi_length is not None else 2 * max_bead_diameter
        self.search_channels = utils.to_list(search_channel)
        self.gui = InteractiveUI() if interactive else None

    def __call__(self, assay: xr.Dataset) -> xr.Dataset:
        if not self.search_channels:
            self.search_channels = assay.channel

        beads = np.empty((0, 3))
        for search_channel in self.search_channels:
            image = utils.to_uint8(assay.image.isel(time=0).sel(channel=search_channel).to_numpy())
            b = utils.find_circles(
                image,
                low_edge_quantile=self.low_edge_quantile,
                high_edge_quantile=self.high_edge_quantile,
                grid_length=20,
                num_iter=self.num_iter,
                min_radius=self.min_bead_radius,
                max_radius=self.max_bead_radius,
                min_dist=self.min_bead_radius,
                min_roundness=self.min_roundness,
                gui=self.gui,
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
                ("mark", "time", "roi_y", "roi_x"),
                da.empty_like(roi, dtype=bool)[:, 0],
            ),
            bg=(
                ("mark", "time", "roi_y", "roi_x"),
                da.empty_like(roi, dtype=bool)[:, 0],
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
        labels = utils.circle_labels(beads.astype(int), assay.sizes["im_y"], assay.sizes["im_x"])

        # Compute the foreground and background masks for all beads.
        # TODO: Don't assume beads don't move across timesteps.
        # Iterate over numpy arrays since indexing over xarrays is slow.
        x = assay.x.isel(time=0).to_numpy()
        y = assay.y.isel(time=0).to_numpy()
        fg = np.empty((num_beads,) + assay.fg.shape[2:], dtype=bool)
        bg = np.empty_like(fg)
        image = assay.image.isel(time=0).sel(channel=self.search_channels).to_numpy()
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

        assay.mg.cache(["roi", "fg", "bg"])
        assay = assay.assign_coords(
            valid=(
                ("mark", "time"),
                np.ones((assay.sizes["mark"], assay.sizes["time"]), dtype=bool),
            ),
        )

        return assay

    @registry.components.register("find_beads")
    def make(
        min_bead_diameter: int,
        max_bead_diameter: int,
        low_edge_quantile: float,
        high_edge_quantile: float,
        num_iter: int,
        min_roundness: float,
        roi_length: int,
        search_channel: str | list[str] | None,
        interactive: bool,
    ):
        return BeadFinder(
            min_bead_diameter=min_bead_diameter,
            max_bead_diameter=max_bead_diameter,
            low_edge_quantile=low_edge_quantile,
            high_edge_quantile=high_edge_quantile,
            num_iter=num_iter,
            min_roundness=min_roundness,
            roi_length=roi_length,
            search_channel=search_channel,
            interactive=interactive,
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


def label_clusters(points, offset, num_clusters, cluster_length, cluster_gap):
    permutation = np.argsort(points)
    points = points[permutation]
    labels = np.ones_like(points, dtype=int) * -1
    # Compute the boundaries of each cluster.
    increments = [offset] + ([cluster_length, cluster_gap] * num_clusters)[:-1]
    boundaries = np.cumsum(increments)
    # Find start/end indexes of points in each cluster & gaps.
    spans = np.searchsorted(points, boundaries)

    # Assign labels to all points that fall within a cluster.
    for i in range(num_clusters):
        labels[spans[2 * i] : spans[2 * i + 1]] = i

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
            print(
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
