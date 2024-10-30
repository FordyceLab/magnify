import napari
import napari.settings
import napari.utils
import numpy as np
import xarray as xr
import numbers
import magnify.utils as utils
import cv2 as cv

from qtpy.QtWidgets import QSlider, QVBoxLayout, QWidget, QLabel, QPushButton
from qtpy.QtCore import Qt, QEventLoop


def continue_button_event(viewer, loop):
    viewer.close()
    loop.quit()


def re_run_button_event(viewer, edges, low_edge_quantile, high_edge_quantile, dx, dy):
    # Excerpt code to re-run from find_circles() in find.py
    grad = np.sqrt(dx**2 + dy**2)
    low_thresh = np.quantile(grad, low_edge_quantile)
    high_thresh = np.quantile(grad, high_edge_quantile)
    edges = cv.Canny(
        dx.astype(np.int16),
        dy.astype(np.int16),
        threshold1=low_thresh,
        threshold2=high_thresh,
        L2gradient=True,
    )
    if "Edges" in viewer.layers:
        viewer.layers.remove("Edges")
    # If the layer doesn't exist, add it as a new layer
    viewer.add_image(edges, name="Edges")


def show_img_button_event(viewer):
    if "Image" in viewer.layers:
        viewer.layers["Image"].visible = not viewer.layers["Image"].visible


def show_edge_button_event(viewer):
    if "Edges" in viewer.layers:
        viewer.layers["Edges"].visible = not viewer.layers["Edges"].visible


def display_edge_detection(img, edges, low_edge_quantile, high_edge_quantile, dx, dy):
    viewer = napari.Viewer()
    loop = QEventLoop()

    # Create and add sliders
    low_edge_slider_widget, low_edge_slider = create_slider(
        "low_edge_quantile", low_edge_quantile, 0.1, 0.0, 1.0
    )
    high_edge_slider_widget, high_edge_slider = create_slider(
        "high_edge_quantile", high_edge_quantile, 0.9, 0.0, 1.0
    )
    viewer.window.add_dock_widget(low_edge_slider_widget, area="right")
    viewer.window.add_dock_widget(high_edge_slider_widget, area="right")

    # Create and add buttons
    rerun_button = create_button(
        "Re-run",
        lambda: re_run_button_event(
            viewer,
            edges,
            low_edge_slider.value() / 100,
            high_edge_slider.value() / 100,
            dx,
            dy,
        ),
    )
    cont_button = create_button("Continue", lambda: continue_button_event(viewer, loop))
    viewer.window.add_dock_widget(rerun_button, area="right")
    viewer.window.add_dock_widget(cont_button, area="right")

    show_img_button = create_button("Show Image", lambda: show_img_button_event(viewer))
    show_edges_button = create_button(
        "Show Edges", lambda: show_edge_button_event(viewer)
    )
    viewer.window.add_dock_widget(show_img_button, area="left")
    viewer.window.add_dock_widget(show_edges_button, area="left")

    viewer.add_image(img, name="Image")
    viewer.add_image(edges, name="Edges")
    viewer.show()
    loop.exec_()

    return edges


def create_button(label_text, click_event):
    widget = QWidget()
    layout = QVBoxLayout()

    button = QPushButton(label_text)
    # Pass button_clicked to on_click_event so we can update it
    button.clicked.connect(click_event)

    layout.addWidget(button)
    widget.setLayout(layout)

    return widget


def create_slider(label_text, value, default_value, min_value, max_value):
    """Create a slider that prints the default, minimium, and maximum values."""
    widget = QWidget()
    layout = QVBoxLayout()

    # Determine if scaling is needed for floats
    is_float = isinstance(min_value, float) or isinstance(max_value, float)
    scale_factor = 100.0 if is_float else 1

    # Scale values for the slider if float
    scaled_min = int(min_value * scale_factor)
    scaled_max = int(max_value * scale_factor)
    scaled_value = int(value * scale_factor)

    # Label to display the slider's current value and range
    label = QLabel(
        f"{label_text} (def={default_value}): {value}\n min={min_value} max={max_value}"
    )

    # Create the slider and set the range and initial value
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(scaled_min)
    slider.setMaximum(scaled_max)
    slider.setValue(scaled_value)

    # Function to update the label when the slider is changed
    def slider_changed(slider_value):
        display_value = slider_value / scale_factor if is_float else slider_value
        label.setText(
            f"{label_text} (def={default_value}): {display_value}\n min={min_value} max={max_value}"
        )

    # Connect slider change to update label
    slider.valueChanged.connect(slider_changed)

    layout.addWidget(label)
    layout.addWidget(slider)
    widget.setLayout(layout)

    return widget, slider
