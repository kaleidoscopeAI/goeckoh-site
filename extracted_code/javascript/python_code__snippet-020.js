"""Get the cursor info - used to get cursor visibility and width

Args:
    std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.
    cursor_info (CONSOLE_CURSOR_INFO): CONSOLE_CURSOR_INFO ctype struct that receives information
        about the console's cursor.

Returns:
      bool: True if the function succeeds, otherwise False.
"""
return bool(_GetConsoleCursorInfo(std_handle, byref(cursor_info)))


