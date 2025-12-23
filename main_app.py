import argparse
import queue
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from src.ui_foam import FoamWidget
from src.neuro_backend import NeuroKernel

# Kivy UI for the Child Interface
class ChildUI(App):
    def __init__(self, ui_queue, **kwargs):
        super().__init__(**kwargs)
        self.ui_queue = ui_queue
        self.foam_widget = FoamWidget()

    def build(self):
        layout = BoxLayout()
        layout.add_widget(self.foam_widget)
        from kivy.clock import Clock
        Clock.schedule_interval(self.process_queue, 1/60.0)
        return layout

    def process_queue(self, dt):
        try:
            while not self.ui_queue.empty():
                message_type, data = self.ui_queue.get_nowait()
                if message_type == "PHY":
                    self.foam_widget.update_physics(data["rms"], data["gcl"])
                elif message_type == "TXT":
                    print(f"UI: {data}")
        except queue.Empty:
            pass

# Textual UI for the Clinician Dashboard
def run_clinician_dashboard(ui_queue):
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, Static
    from textual.reactive import reactive

    class ClinicianDashboard(App):
        BINDINGS = [("q", "quit", "Quit")]
        
        rms = reactive(0.0)
        gcl = reactive(0.0)
        ent = reactive(0.0)
        last_exchange = reactive("")

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static(id="gcl_display")
            yield Static(id="rms_display")
            yield Static(id="ent_display")
            yield Static(id="txt_display")
            yield Footer()

        def on_mount(self) -> None:
            self.set_interval(1/30.0, self.update_displays)
            self.set_interval(1/10.0, self.process_queue)
        
        def process_queue(self) -> None:
            try:
                while not ui_queue.empty():
                    msg_type, data = ui_queue.get_nowait()
                    if msg_type == "PHY":
                        self.rms = data["rms"]
                        self.gcl = data["gcl"]
                        self.ent = data["ent"]
                    elif msg_type == "TXT":
                        self.last_exchange = data
            except queue.Empty:
                pass

        def update_displays(self) -> None:
            self.query_one("#gcl_display").update(f"GCL: {self.gcl:.3f}")
            self.query_one("#rms_display").update(f"RMS: {self.rms:.3f}")
            self.query_one("#ent_display").update(f"ENT: {self.ent:.3f}")
            self.query_one("#txt_display").update(f"TXT: {self.last_exchange}")


    app = ClinicianDashboard()
    app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["clinician", "child"], required=True)
    args = parser.parse_args()

    ui_queue = queue.Queue()

    kernel = NeuroKernel(ui_queue=ui_queue)
    kernel.start()

    if args.mode == "child":
        ChildUI(ui_queue).run()
    elif args.mode == "clinician":
        run_clinician_dashboard(ui_queue)
