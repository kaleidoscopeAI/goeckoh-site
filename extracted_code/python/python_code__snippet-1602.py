import ctypes
from ctypes import LibraryLoader

if sys.platform == "win32":
    windll = LibraryLoader(ctypes.WinDLL)
else:
    windll = None
    raise ImportError("Not windows")

from pip._vendor.rich._win32_console import (
    ENABLE_VIRTUAL_TERMINAL_PROCESSING,
    GetConsoleMode,
    GetStdHandle,
    LegacyWindowsError,
)

