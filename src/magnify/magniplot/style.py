from holoviews import opts
import holoviews as hv


styles = {
    "whitegrid": [
        opts.Layout(shared_axes=False),
        opts.GridSpace(shared_xaxis=False, shared_yaxis=False),
        opts.Image(invert_yaxis=True, axiswise=True),
    ]
}


def set_style(name="whitegrid"):
    opts.defaults(*styles[name])
