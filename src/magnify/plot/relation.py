import holoviews as hv
import numpy as np
import plotly.express as px
import scipy.stats
import xarray as xr

from magnify import utils
from magnify.plot.ndplot import ndplot


def relplot(
    assay: xr.Dataset,
    x="time.seconds",
    y="tag_intensity",
    fit_method="linear",
    grid=None,
    hue=None,
    slider=None,
    **kwargs,
):
    if grid is None and slider is None:
        if y == "mark_intensity":
            slider = ["channel", "tag"]
        else:
            slider = ["channel", "tag"]
    if hue is None:
        hue = ["mark", "mark_row", "mark_col"]
        hue = [h for h in hue if h in assay.dims]

    fit_func = {"linear": linear_fit, "quadratic": quadratic_fit, "exp_linear": exp_linear_fit}[
        fit_method
    ]

    def relfunc(assay: xr.Dataset, **kwargs):
        assay = utils.sel_tag(assay, assay.tag)
        assay = assay.where(assay.valid, drop=True)
        overlays = []
        if assay.sizes[hue] == 0:
            return hv.Overlay([hv.Points((0, 0))])
        for name, group in assay.groupby(hue):
            l = str((int(group.mark_row), int(group.mark_col)))
            x_v = group[x].to_numpy()
            y_v = group[y].to_numpy()
            points = hv.Points((x_v, y_v)).opts(**kwargs)
            curve = hv.Curve((x_v, fit_func(x_v, y_v)))
            overlays.append(points * curve)

        return hv.Overlay(overlays)

    return ndplot(assay, relfunc, grid=grid, slider=slider, **kwargs)


def linear_fit(x, y):
    m, b, _, _, _ = scipy.stats.linregress(x, y)
    return m * x + b


def quadratic_fit(x, y):
    X = np.column_stack((x**2, x, np.ones(len(x))))
    theta = np.linalg.lstsq(X, y, rcond=None)[0]
    return X @ theta


def exp_linear_fit(x, y):
    if np.isnan(y).any():
        return 0.0 * x
    x = np.asarray(x)
    y = np.asarray(y)

    def func(x, a, k, m, b):
        return a * np.exp(-k * x) - m * x + b

    def jac(x, a, k, m, b):
        return np.column_stack(
            [
                np.exp(-k * x),
                -a * x * np.exp(-k * x),
                -x,
                np.ones_like(x),
            ]
        )

    m, b, _, _, _ = scipy.stats.linregress(x, np.log(y - y.min() + 1e-5))
    theta, _ = scipy.optimize.curve_fit(
            func, x, y, p0=(max(y[0] - y[-1], 0), 2e-4, max((y[-5] - y[-1]) / 10000, 0), y[0]),
            bounds=((max((y[0] - y[-1]) / 2, 0), 0.0, 0.0, 0.0), (np.inf, 0.1, np.inf, np.inf)),
            jac=jac)
    return func(x, *theta)
