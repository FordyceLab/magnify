import cv2 as cv
import napari
import numpy as np
from qtpy.QtCore import QEventLoop, Qt
from qtpy.QtWidgets import QLabel, QPushButton, QSlider, QVBoxLayout, QWidget


class EdgeDetectionUI:
    def __init__(self, img, edges, low_edge_quantile, high_edge_quantile, dx, dy):
        self.img = img
        self.edges = edges
        self.low_edge_quantile = low_edge_quantile
        self.high_edge_quantile = high_edge_quantile
        self.dx = dx
        self.dy = dy

        self.viewer = napari.Viewer()
        self.loop = QEventLoop()
        self.setup_ui()

    def setup_ui(self):
        # Add sliders
        self.low_edge_slider_widget, self.low_edge_slider = self.create_slider(
            "low_edge_quantile", self.low_edge_quantile, 0.1, 0.0, 1.0
        )
        self.high_edge_slider_widget, self.high_edge_slider = self.create_slider(
            "high_edge_quantile", self.high_edge_quantile, 0.9, 0.0, 1.0
        )
        self.viewer.window.add_dock_widget(self.low_edge_slider_widget, area="right")
        self.viewer.window.add_dock_widget(self.high_edge_slider_widget, area="right")

        # Add buttons
        rerun_button = self.create_button("Re-run", self.re_run_event)
        cont_button = self.create_button("Continue", self.continue_event)
        self.viewer.window.add_dock_widget(rerun_button, area="right")
        self.viewer.window.add_dock_widget(cont_button, area="right")

        show_img_button = self.create_button("Show Image", self.toggle_image_visibility)
        show_edges_button = self.create_button(
            "Show Edges", self.toggle_edges_visibility
        )
        self.viewer.window.add_dock_widget(show_img_button, area="left")
        self.viewer.window.add_dock_widget(show_edges_button, area="left")

        # Add layers to viewer
        self.viewer.add_image(self.img, name="Image")
        self.viewer.add_image(self.edges, name="Edges")
        self.viewer.show()

    def re_run_event(self):
        grad = np.sqrt(self.dx**2 + self.dy**2)
        low_thresh = np.quantile(grad, self.low_edge_slider.value() / 100)
        high_thresh = np.quantile(grad, self.high_edge_slider.value() / 100)
        self.edges = cv.Canny(
            self.dx.astype(np.int16),
            self.dy.astype(np.int16),
            threshold1=low_thresh,
            threshold2=high_thresh,
            L2gradient=True,
        )
        if "Edges" in self.viewer.layers:
            self.viewer.layers["Edges"].data = self.edges

    def continue_event(self):
        self.viewer.close()
        self.loop.quit()

    def toggle_image_visibility(self):
        if "Image" in self.viewer.layers:
            self.viewer.layers["Image"].visible = not self.viewer.layers[
                "Image"
            ].visible

    def toggle_edges_visibility(self):
        if "Edges" in self.viewer.layers:
            self.viewer.layers["Edges"].visible = not self.viewer.layers[
                "Edges"
            ].visible

    @staticmethod
    def create_button(label_text, click_event):
        widget = QWidget()
        layout = QVBoxLayout()

        button = QPushButton(label_text)
        button.clicked.connect(click_event)

        layout.addWidget(button)
        widget.setLayout(layout)

        return widget

    @staticmethod
    def create_slider(label_text, value, default_value, min_value, max_value):
        widget = QWidget()
        layout = QVBoxLayout()

        is_float = isinstance(min_value, float) or isinstance(max_value, float)
        scale_factor = 100.0 if is_float else 1

        scaled_min = int(min_value * scale_factor)
        scaled_max = int(max_value * scale_factor)
        scaled_value = int(value * scale_factor)

        label = QLabel(
            f"{label_text} (def={default_value}): {value}\n min={min_value} max={max_value}"
        )

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(scaled_min)
        slider.setMaximum(scaled_max)
        slider.setValue(scaled_value)

        def slider_changed(slider_value):
            display_value = slider_value / scale_factor if is_float else slider_value
            label.setText(
                f"{label_text} (def={default_value}): {display_value}\n min={min_value} max={max_value}"
            )

        slider.valueChanged.connect(slider_changed)

        layout.addWidget(label)
        layout.addWidget(slider)
        widget.setLayout(layout)

        return widget, slider


def display_edge_detection(img, edges, low_edge_quantile, high_edge_quantile, dx, dy):
    ui = EdgeDetectionUI(img, edges, low_edge_quantile, high_edge_quantile, dx, dy)
    ui.loop.exec_()
    return ui.edges
