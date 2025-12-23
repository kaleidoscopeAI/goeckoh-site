from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog

class MainApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "DeepPurple"
        layout = MDBoxLayout(orientation="vertical", padding=40, spacing=20)
        layout.add_widget(MDLabel(text=f"[color=#ffffff]Jackson’s Companion\nAlways Listening[/color]", halign="center", font_style="H3", markup=True))
        layout.add_widget(MDLabel(text="Status: Active", halign="center"))
        layout.add_widget(MDFlatButton(text="Close", on_release=lambda x: self.stop()))
        return layout

