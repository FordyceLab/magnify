from qtpy.QtCore import QEventLoop
import magicgui
import napari
import napari.settings

class InteractiveUI:
    def __init__(self):
        self.viewer = napari.Viewer()
        self.event_loop = QEventLoop()
        self.widget = None

    def continue_event(self, last):
        self.viewer.layers.clear()
        self.viewer.window.remove_dock_widget(self.widget)
        if last:
            self.viewer.close()

        self.event_loop.quit()

    def run_widget(self, func, auto_call=False, last=False):
        # Make the widget.
        widget_func = magicgui.magicgui(func, auto_call=auto_call)
        self.widget = self.viewer.window.add_dock_widget(widget_func, area="right")
        # Setup the continue button.
        continue_btn = magicgui.widgets.PushButton(value=True, text="Continue")
        continue_btn.changed.connect(lambda: self.continue_event(last=last))
        widget_func.append(continue_btn)
        # Call the widget function to initialize it in the viewer.
        widget_func()

        # If we are in a notebook we don't want napari to integrate with the
        # ipython loop since we want it to run in qt.
        settings = napari.settings.get_settings().application
        ipy = settings.ipy_interactive
        settings.ipy_interactive = False
        # Run the event loop.
        self.viewer.show()
        self.event_loop.exec_()
        # Reset ipy_interactive to its previous value.
        settings.ipy_interactive = ipy

        return widget_func()
