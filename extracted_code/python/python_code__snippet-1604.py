def get_windows_console_features() -> WindowsConsoleFeatures:
    """Get windows console features.

    Returns:
        WindowsConsoleFeatures: An instance of WindowsConsoleFeatures.
    """
    handle = GetStdHandle()
    try:
        console_mode = GetConsoleMode(handle)
        success = True
    except LegacyWindowsError:
        console_mode = 0
        success = False
    vt = bool(success and console_mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    truecolor = False
    if vt:
        win_version = sys.getwindowsversion()
        truecolor = win_version.major > 10 or (
            win_version.major == 10 and win_version.build >= 15063
        )
    features = WindowsConsoleFeatures(vt=vt, truecolor=truecolor)
    return features


