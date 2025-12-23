"""Set the cursor info - used for adjusting cursor visibility and width

Args:
    std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.
    cursor_info (CONSOLE_CURSOR_INFO): CONSOLE_CURSOR_INFO ctype struct containing the new cursor info.

Returns:
      bool: True if the function succeeds, otherwise False.
"""
return bool(_SetConsoleCursorInfo(std_handle, byref(cursor_info)))


