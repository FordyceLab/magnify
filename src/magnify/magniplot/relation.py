import holoviews as hv
import scipy.stats
import xarray as xr

from magnify.magniplot.ndplot import ndplot


def relplot(
    assay: xr.Dataset,
    x="time.seconds",
    y="intensity",
    fit_method="linear",
    grid=None,
    slider=None,
    **kwargs,
):
    if grid is None and slider is None:
        slider = ["channel", "mark", "mark_row", "mark_col"]

    fit_func = {"linear": linear_fit}[fit_method]

    def relfunc(assay: xr.Dataset, **kwargs):
        points = hv.Points((assay[x], assay[y])).opts(**kwargs)
        curve = hv.Curve((assay[x], fit_func(assay[x], assay[y])))
        return points * curve

    return ndplot(assay, relfunc, grid=grid, slider=slider, **kwargs)


def linear_fit(x, y):
    m, b, _, _, _ = scipy.stats.linregress(x, y)
    return m * x + b
