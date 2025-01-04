import napari
from qtpy.QtCore import QEventLoop
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

from magnify import find

viewer = None


class EdgeDetectionUI:
    def __init__(self, img, edges):
        self.img = img
        self.edges = edges
        self.loop = QEventLoop()
        self.setup_ui()

    def setup_ui(self):
        viewer.window.add_dock_widget(find.compute_edges)

        # Add buttons
        cont_button = self.create_button("Continue", self.continue_event)
        viewer.window.add_dock_widget(cont_button, area="right")

        show_img_button = self.create_button("Show Image", self.toggle_image_visibility)
        show_edges_button = self.create_button("Show Edges", self.toggle_edges_visibility)
        viewer.window.add_dock_widget(show_img_button, area="left")
        viewer.window.add_dock_widget(show_edges_button, area="left")

        # Add layers to viewer
        viewer.add_image(self.img, name="Image")
        viewer.add_image(self.edges, name="Edges")
        viewer.show()

    def continue_event(self):
        self.loop.quit()
        viewer.layers.clear()
        for dock_widget in list(viewer.window._dock_widgets.values()):
            viewer.window.remove_dock_widget(dock_widget)

    def toggle_image_visibility(self):
        if "Image" in viewer.layers:
            viewer.layers["Image"].visible = not viewer.layers["Image"].visible

    def toggle_edges_visibility(self):
        if "Edges" in viewer.layers:
            viewer.layers["Edges"].visible = not viewer.layers["Edges"].visible

    @staticmethod
    def create_button(label_text, click_event):
        widget = QWidget()
        layout = QVBoxLayout()

        button = QPushButton(label_text)
        button.clicked.connect(click_event)

        layout.addWidget(button)
        widget.setLayout(layout)

        return widget


def display_ui(img, edges, dx, dy, vis_pipe):
    if vis_pipe:
        viewer = napari.Viewer
        ui = EdgeDetectionUI(img, edges)
        find.compute_edges.dx.value = dx
        find.compute_edges.dy.value = dy
        ui.loop.exec_()
        return ui.edges
    return edges


@find.compute_edges.called.connect
def update_edges_layer(edges_tuple):
    edges, meta = edges_tuple
    if "Edges" in viewer.layers:
        viewer.layers["Edges"].data = edges
    else:
        viewer.add_image(edges, name="Edges")
