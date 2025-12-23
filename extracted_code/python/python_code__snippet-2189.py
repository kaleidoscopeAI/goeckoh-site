# Kivy properties automatically update UI when changed
gcl_text = StringProperty("1.00")
state_text = StringProperty("INIT")
stress_text = StringProperty("0.00")
last_response = StringProperty("Waiting for input...")
heart_color = ColorProperty((0, 1, 1, 1))

def build(self):
    self.theme_cls.theme_style = "Dark"
    self.theme_cls.primary_palette = "Cyan"

    # Init Systems
    self.heart_logic = CrystallineHeart()
    self.audio = AudioBridge()

    # Physics loop (Clock)
    Clock.schedule_interval(self.update_physics, 0.1)

    return Builder.load_string(KV)

def submit_input(self, field):
    text = field.text
    if not text: return

    # Process Logic (Heart/Mirror/Gating)
    resp, metrics = self.heart_logic.process_input(text)

    # Trigger Audio (Rust Kernel via Bridge)
    # We send the stress level (1 - GCL) as 'arousal'
    arousal = 1.0 - metrics.gcl
    self.audio.enqueue_response(resp, arousal)

    # Update GUI logic (text)
    self.last_response = resp

    field.text = "" # Clear input

def update_physics(self, dt):
    """Called 10 times per second to animate the heart."""
    # Idle processing (natural decay)
    _, metrics = self.heart_logic.process_input("")

    # Update Reactive Properties
    self.gcl_text = f"{metrics.gcl:.2f}"
    self.stress_text = f"{metrics.stress:.2f}"
    self.state_text = metrics.mode_label

    # Animate Crystal Color
    self.heart_color = metrics.gui_color

def trigger_panic(self):
    """Kill Switch"""
    self.heart_logic.nodes = [0.0] * 1024 # Hard reset lattice
    self.last_response = "SYSTEM RESET. BREATHING..."

def on_stop(self):
    self.audio.running = False

