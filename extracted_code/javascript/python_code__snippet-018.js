"""Set the colour attributes for all text written after this function is called.

Args:
    std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.
    attributes (int): Integer value representing the foreground and background colours.


Returns:
    bool: True if the attribute was set successfully, otherwise False.
"""
return bool(_SetConsoleTextAttribute(std_handle, attributes))


