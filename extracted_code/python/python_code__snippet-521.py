import contextlib
import ssl
import typing
from ctypes import WinDLL  # type: ignore
from ctypes import WinError  # type: ignore
from ctypes import (
    POINTER,
    Structure,
    c_char_p,
    c_ulong,
    c_void_p,
    c_wchar_p,
    cast,
    create_unicode_buffer,
    pointer,
    sizeof,
