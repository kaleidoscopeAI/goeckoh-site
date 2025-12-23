"""
Evaluate a PEP 508 environment marker.
Return a boolean indicating the marker result in this environment.
Raise SyntaxError if marker is invalid.

This implementation uses the 'pyparsing' module.
"""
try:
    marker = packaging.markers.Marker(text)
    return marker.evaluate()
except packaging.markers.InvalidMarker as e:
    raise SyntaxError(e) from e


