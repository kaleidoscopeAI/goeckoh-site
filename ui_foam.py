from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse
from kivy.clock import Clock
from kivy.properties import NumericProperty

class FoamWidget(Widget):
    rms = NumericProperty(0.0)
    gcl = NumericProperty(1.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Clock.schedule_interval(self._update_canvas, 1/30.0)

    def update_physics(self, rms, gcl):
        self.rms = rms
        self.gcl = gcl

    def _update_canvas(self, dt):
        self.canvas.clear()
        with self.canvas:
            intensity = min(self.rms * 10.0, 1.0)
            calm = self.gcl
            
            # Bouba/Kiki Color Logic: Calm = Blue/Green, Stress = Red/Spiky
            Color(intensity, calm, 1.0 - calm, 0.8)
            
            d = 50 + intensity * 150
            x = self.center_x - d/2
            y = self.center_y - d/2
            Ellipse(pos=(x, y), size=(d, d))
