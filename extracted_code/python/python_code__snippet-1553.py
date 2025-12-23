"""Coordinates in the Windows Console API are (y, x), not (x, y).
This class is intended to prevent that confusion.
Rows and columns are indexed from 0.
This class can be used in place of wintypes._COORD in arguments and argtypes.
"""

row: int
col: int

@classmethod
def from_param(cls, value: "WindowsCoordinates") -> COORD:
    """Converts a WindowsCoordinates into a wintypes _COORD structure.
    This classmethod is internally called by ctypes to perform the conversion.

    Args:
        value (WindowsCoordinates): The input coordinates to convert.

    Returns:
        wintypes._COORD: The converted coordinates struct.
    """
    return COORD(value.col, value.row)


