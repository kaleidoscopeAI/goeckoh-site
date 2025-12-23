"""Set the position of the cursor in the console screen

Args:
    std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.
    coords (WindowsCoordinates): The coordinates to move the cursor to.

Returns:
    bool: True if the function succeeds, otherwise False.
"""
return bool(_SetConsoleCursorPosition(std_handle, coords))


