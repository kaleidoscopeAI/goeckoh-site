          bool: True if the function succeeds, otherwise False.
    """
    return bool(_GetConsoleCursorInfo(std_handle, byref(cursor_info)))


