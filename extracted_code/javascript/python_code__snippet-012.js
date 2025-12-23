          bool: True if the function succeeds, otherwise False.
    """
    return bool(_SetConsoleCursorInfo(std_handle, byref(cursor_info)))


