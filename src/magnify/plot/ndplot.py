from typing import Callable
import functools
import itertools
import math

import holoviews as hv
import xarray as xr

import magnify.utils as utils


def ndplot(
    assay,
    plot_function,
    grid: str | list[str] | None = None,
    slider: str | list[str] | None = None,
    **kwargs,
):
    assay = utils.to_explicit_coords(assay)
    grid = [dim for dim in utils.to_list(grid) if dim in assay.indexes]
    slider = [dim for dim in utils.to_list(slider) if dim in assay.indexes]

    def to_holomap(subassay):
        if slider:
            slider_coords = [subassay[dim].values for dim in slider]
            return hv.HoloMap(
                {
                    idx: plot_function(subassay.sel(dict(zip(slider, idx))), **kwargs)
                    for idx in itertools.product(*slider_coords)
                },
                kdims=slider,
            )
        else:
            # We don't have sliders so just return the base plot.
            return plot_function(subassay, **kwargs)

    if len(grid) > 2:
        raise ValueError(f"Cannot plot more than 2 grid dimensions. Got {grid}.")
    elif len(grid) == 2:
        grid_coords0 = assay[grid[0]].values
        grid_coords1 = assay[grid[1]].values
        return hv.GridSpace(
            {
                (c0, c1): to_holomap(assay.sel({grid[0]: c0, grid[1]: c1}))
                for c0 in grid_coords0
                for c1 in grid_coords1
            }
        )
    elif len(grid) == 1:
        # We only have a single index so display the resulting image in a layout with
        # an approximately equal number of rows and columns.
        grid_coords = assay[grid[0]].values
        num_cols = math.ceil(len(grid_coords) ** 0.5)
        return hv.Layout([to_holomap(assay.sel({grid[0]: c})) for c in grid_coords]).cols(num_cols)
    else:
        return to_holomap(assay)
