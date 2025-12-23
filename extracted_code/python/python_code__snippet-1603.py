# Fallback if we can't load the Windows DLL
def get_windows_console_features() -> WindowsConsoleFeatures:
    features = WindowsConsoleFeatures()
    return features

