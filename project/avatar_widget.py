# gui/avatar_widget.py
from __future__ import annotations
from typing import Optional
import numpy as np
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse
from events import AvatarFrame

class AvatarWidget(Widget):
    """
    Simple 2D visualization of the Crystal Avatar.
    - Projects 3D positions in [-1, 1] to screen coordinates.
    - Uses node color (r,g,b) and size for circle color and radius.
    - Downsamples to a configurable number of nodes for performance.
    """
    def __init__(self, max_nodes: int = 1200, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_nodes = max_nodes
        self._last_frame: Optional[AvatarFrame] = None

    def update_from_frame(self, frame: AvatarFrame) -> None:
        """Receives a new frame and triggers a redraw."""
        self._last_frame = frame
        self._redraw()

    def on_size(self, *args) -> None:
        """When the widget resizes, re-render the last frame if available."""
        if self._last_frame is not None:
            self._redraw()

    def on_pos(self, *args) -> None:
        """Re-render on position change so coordinates stay aligned."""
        if self._last_frame is not None:
            self._redraw()

    def _redraw(self) -> None:
        """Clears the canvas and draws the avatar nodes from the last frame."""
        frame = self._last_frame
        if frame is None or frame.positions.size == 0:
            self.canvas.clear()
            return

        positions = frame.positions
        colors = frame.colors
        sizes = frame.sizes

        # Downsample for rendering performance if needed
        n = positions.shape[0]
        if n > self.max_nodes:
            idx = np.linspace(0, n - 1, self.max_nodes, dtype=int)
            positions = positions[idx]
            colors = colors[idx]
            sizes = sizes[idx]

        w, h = float(self.width), float(self.height)
        if w <= 1 or h <= 1:
            self.canvas.clear()
            return

        # Map positions from [-1, 1] to [0, w] and [0, h]
        pos_xy = positions[:, :2]
        x = (pos_xy[:, 0] + 1.0) * 0.5 * w
        y = (pos_xy[:, 1] + 1.0) * 0.5 * h

        # Radius based on size (and a minimum so dots are visible)
        base_radius = 2.0
        radii = base_radius + 6.0 * np.clip(sizes.astype(float), 0.0, 1.0)

        # Clear and redraw
        self.canvas.clear()
        with self.canvas:
            for i in range(len(x)):
                xi, yi, ri = float(x[i]), float(y[i]), float(radii[i])
                r, g, b = float(colors[i, 0]), float(colors[i, 1]), float(colors[i, 2])
                
                Color(np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1), 0.85)
                Ellipse(
                    pos=(xi - ri, yi - ri),
                    size=(2.0 * ri, 2.0 * ri),
                )
