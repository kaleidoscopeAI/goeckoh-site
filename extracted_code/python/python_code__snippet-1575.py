"""Render segments to Jupyter."""
html = _render_segments(segments)
jupyter_renderable = JupyterRenderable(html, text)
try:
    from IPython.display import display as ipython_display

    ipython_display(jupyter_renderable)
except ModuleNotFoundError:
    # Handle the case where the Console has force_jupyter=True,
    # but IPython is not installed.
    pass


