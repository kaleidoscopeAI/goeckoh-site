from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.properties import StringProperty, ColorProperty
import threading

# Import internal modules
from goeckoh.heart.logic_core import CrystallineHeart
from goeckoh.audio.audio_bridge import AudioSystem

# --- GUI LAYOUT (KV Language) ---
KV = '''
MDScreen:
    md_bg_color: 0.05, 0.05, 0.1, 1

    MDBoxLayout:
        orientation: "vertical"
        padding: 20
        spacing: 20

        # HEADER
        MDLabel:
            text: "NEURO-ACOUSTIC EXOCORTEX"
            halign: "center"
            theme_text_color: "Custom"
            text_color: 0.8, 0.9, 1, 1
            font_style: "H5"
            size_hint_y: None
            height: 60

        # VISUALIZER (THE HEART)
        MDAnchorLayout:
            size_hint_y: 1.5
            
            MDCard:
                size_hint: None, None
                size: "250dp", "250dp"
                radius: [125,]
                md_bg_color: 0.1, 0.1, 0.15, 1
                elevation: 4
                
                MDIcon:
                    icon: "diamond-stone"
                    halign: "center"
                    theme_text_color: "Custom"
                    text_color: app.heart_color
                    font_size: "150sp"
                    pos_hint: {"center_x": 0.5, "center_y": 0.5}

        # METRICS DISPLAY
        MDGridLayout:
            cols: 3
            size_hint_y: None
            height: 60
            spacing: 10

            MDCard:
                orientation: "vertical"
                padding: 10
                md_bg_color: 0.15, 0.15, 0.2, 1
                MDLabel:
                    text: "GCL"
                    halign: "center"
                    theme_text_color: "Secondary"
                MDLabel:
                    text: app.gcl_text
                    halign: "center"
                    theme_text_color: "Custom"
                    text_color: app.heart_color
            
            MDCard:
                orientation: "vertical"
                padding: 10
                md_bg_color: 0.15, 0.15, 0.2, 1
                MDLabel:
                    text: "STATE"
                    halign: "center"
                    theme_text_color: "Secondary"
                MDLabel:
                    text: app.state_text
                    halign: "center"
                    bold: True

            MDCard:
                orientation: "vertical"
                padding: 10
                md_bg_color: 0.15, 0.15, 0.2, 1
                MDLabel:
                    text: "STRESS"
                    halign: "center"
                    theme_text_color: "Secondary"
                MDLabel:
                    text: app.stress_text
                    halign: "center"

        # RESPONSE TEXT
        MDCard:
            size_hint_y: None
            height: 100
            padding: 20
            md_bg_color: 0.1, 0.1, 0.1, 0.5
            MDLabel:
                text: app.last_response
                halign: "center"
                theme_text_color: "Primary"
                font_style: "Subtitle1"

        # CONTROLS
        MDBoxLayout:
            spacing: 10
            size_hint_y: None
            height: 50

            MDTextField:
                id: input_field
                hint_text: "Enter thought..."
                mode: "fill"
                text_color_normal: 1, 1, 1, 1
                fill_color_normal: 0.2, 0.2, 0.2, 1
            
            MDRaisedButton:
                text: "PROCESS"
                on_release: app.submit_input(input_field)
            
            MDFloatingActionButton:
                icon: "power-standby"
                md_bg_color: 1, 0, 0, 1
                on_release: app.trigger_panic()
'''

class MainApp(MDApp):
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
        self.audio = AudioSystem()
        
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

if __name__ == "__main__":
    MainApp().run()
