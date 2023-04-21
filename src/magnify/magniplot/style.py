from holoviews import opts
import holoviews as hv


styles = {
    "whitegrid": [
        opts.Layout(shared_axes=False),
        opts.GridSpace(shared_xaxis=False, shared_yaxis=False),
        opts.Image(invert_yaxis=True, axiswise=True, cmap="viridis"),
    ]
}


def set_style(name="whitegrid"):
    hv.extension("bokeh")
    opts.defaults(*styles[name])
