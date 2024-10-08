from __future__ import annotations

import plotly.io as pio


def set_style(name="whitegrid"):
    pio.renderers["jupyterlab"].config["scrollZoom"] = True
