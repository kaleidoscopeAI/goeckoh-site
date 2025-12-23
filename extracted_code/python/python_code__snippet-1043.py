import os
import platform
import socket
import ssl
import typing

import _ssl  # type: ignore[import]

from ._ssl_constants import (
    _original_SSLContext,
    _original_super_SSLContext,
    _truststore_SSLContext_dunder_class,
    _truststore_SSLContext_super_class,
