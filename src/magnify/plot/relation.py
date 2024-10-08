from __future__ import annotations

import numpy as np
import plotly.express as px
import scipy.stats
import xarray as xr

from magnify.plot.ndplot import ndplot


def relplot(
    assay: xr.Dataset,
    x="time",
    y="intensity",
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

    def relfunc(assay: xr.Dataset, **kwargs):
        assay = assay.where(assay.valid, drop=True)
        return px.scatter(assay.to_dataframe().reset_index(), x=x, y=y, color=hue).data

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
        func,
        x,
        y,
        p0=(max(y[0] - y[-1], 0), 2e-4, max((y[-5] - y[-1]) / 10000, 0), y[0]),
        bounds=((max((y[0] - y[-1]) / 2, 0), 0.0, 0.0, 0.0), (np.inf, 0.1, np.inf, np.inf)),
        jac=jac,
    )
    return func(x, *theta)
