from holoviews import opts
import holoviews as hv


styles = {
    "whitegrid": [
        opts.Points(
            autorange="y", framewise=True, frame_width=500, frame_height=500, show_grid=True
        ),
        opts.Curve(autorange="y", framewise=True, show_grid=True),
        opts.Overlay(autorange="y", shared_axes=False, show_grid=True),
        opts.Layout(axiswise=True, shared_axes=False),
        opts.GridSpace(shared_xaxis=False, shared_yaxis=False),
        opts.Image(
            invert_yaxis=True, axiswise=True, cmap="viridis", frame_width=500, frame_height=500
        ),
    ]
}


def set_style(name="whitegrid"):
    hv.extension("bokeh")
    opts.defaults(*styles[name])
