if HEADLESS:
    raise ImportError("Headless mode - skip Kivy imports")
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
KIVY_AVAILABLE = True
