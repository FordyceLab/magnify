from qtpy.QtCore import QEventLoop
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget


class ViewerUI:
    def __init__(self, gui, img, caller):
        # self.edges = edges
        self.viewer = gui
        self.loop = QEventLoop()
        self.caller = caller

        # UI Setup
        self.viewer.window.add_dock_widget(self.caller, area="right")
        if self.caller is not None:
            self.caller.called.connect(self.update_layer)

        # Add the continue button
        cont_button = self.create_button("Continue", self.continue_event)
        self.viewer.window.add_dock_widget(cont_button, area="right")

        # Add layers to the self.viewer
        self.viewer.add_image(img, name="Image")
        # self.viewer.add_image(self.edges, name="Edges")
        self.viewer.add_image(self.caller.value, name="Edges")
        self.viewer.show()

    def update_layer(self, layer_dict):
        layer, value = next(iter(layer_dict.items()))
        if layer in self.viewer.layers:
            self.viewer.layers[layer].data = value
        else:
            self.viewer.add_image(value, name=layer)

    def continue_event(self):
        self.loop.quit()
        self.viewer.layers.clear()
        for dock_widget in list(self.viewer.window._dock_widgets.values()):
            self.viewer.window.remove_dock_widget(dock_widget)

    @staticmethod
    def create_button(label_text, click_event):
        widget = QWidget()
        layout = QVBoxLayout()

        button = QPushButton(label_text)
        button.clicked.connect(click_event)

        layout.addWidget(button)
        widget.setLayout(layout)

        return widget


def display_ui(gui, img, dx, dy, widget=None):
    ui = ViewerUI(gui, img, widget)
    widget.dx.bind(dx)
    widget.dy.bind(dy)
    ui.loop.exec_()
    return widget.value
