if hasattr(ctypes, "windll"):
    return get_win_folder_via_ctypes
try:
    import winreg  # noqa: F401
except ImportError:
    return get_win_folder_from_env_vars
else:
    return get_win_folder_from_registry


