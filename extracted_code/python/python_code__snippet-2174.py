# Reactive properties
heart_color = ColorProperty((0, 1, 1, 1))
gcl_status = StringProperty("SYSTEM ONLINE")
response_text = StringProperty("Waiting...")

def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.heart = None
    self.audio = None

def build(self):
    self.theme_cls.theme_style = "Dark"
    self.theme_cls.primary_palette = "Cyan"

    # Subsystem Initialization
    self.heart = CrystallineHeart()
    self.audio = AudioBridge()

    # Start Physics Loop (15 FPS)
    Clock.schedule_interval(self.tick_physics, 0.066)

    return Builder.load_string(KV_LAYOUT)

def submit_thought(self, widget):
    text = widget.text
    if not text: return

    # Clear input immediately
    widget.text = ""

    # Execute Logic
    resp, metrics = self.heart.process_input(text)

    # Send to Audio Bridge
    stress_val = 1.0 - metrics.gcl
    self.audio.trigger(resp, stress_val)

    # Update UI
    self.response_text = resp

def tick_physics(self, _dt):
    # Process Idle (Time step)
    _, metrics = self.heart.process_input("")

    # Visual Feedback
    self.heart_color = metrics.gui_color
    self.gcl_status = f"GCL: {metrics.gcl:.2f} | {metrics.mode_label}"

