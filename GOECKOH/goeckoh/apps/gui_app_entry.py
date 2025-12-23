from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.properties import StringProperty, ColorProperty

# Architecture Imports
from goeckoh.heart.logic_core import CrystallineHeart
from goeckoh.audio.audio_bridge import AudioBridge

# Kivy Layout - Strict syntax
KV_LAYOUT = '''
MDScreen:
    md_bg_color: 0.05, 0.05, 0.08, 1.0

    MDBoxLayout:
        orientation: "vertical"
        padding: "20dp"
        spacing: "20dp"

        MDLabel:
            text: "NEURO-ACOUSTIC EXOCORTEX"
            halign: "center"
            theme_text_color: "Custom"
            text_color: 0.4, 0.8, 1.0, 1.0
            font_style: "H5"
            bold: True
            size_hint_y: None
            height: "50dp"

        MDCard:
            size_hint: None, None
            size: "260dp", "260dp"
            radius: [130, 130, 130, 130]
            md_bg_color: 0.08, 0.08, 0.12, 1.0
            elevation: 2
            pos_hint: {"center_x": 0.5}
            
            MDIcon:
                icon: "diamond-stone"
                halign: "center"
                theme_text_color: "Custom"
                text_color: app.heart_color
                font_size: "140sp"
                pos_hint: {"center_x": 0.5, "center_y": 0.5}

        MDBoxLayout:
            orientation: "vertical"
            size_hint_y: None
            height: "100dp"
            spacing: "5dp"

            MDLabel:
                text: app.gcl_status
                halign: "center"
                font_style: "H6"
                theme_text_color: "Custom"
                text_color: 1, 1, 1, 0.9
            
            MDLabel:
                text: app.response_text
                halign: "center"
                font_style: "Subtitle1"
                theme_text_color: "Custom"
                text_color: 0.7, 0.7, 0.7, 1.0

        MDTextField:
            id: user_input
            hint_text: "Input thought stream..."
            mode: "fill"
            fill_color_normal: 0.1, 0.1, 0.15, 1.0
            text_color_normal: 1, 1, 1, 1.0
            on_text_validate: app.submit_thought(self)

        Widget:
            size_hint_y: 1
'''

class MainSystemApp(MDApp):
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

if __name__ == "__main__":
    MainSystemApp().run()
