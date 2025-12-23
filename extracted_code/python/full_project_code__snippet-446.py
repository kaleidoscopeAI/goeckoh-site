import os.path
import socket  # noqa: F401

from pip._vendor.urllib3.exceptions import ClosedPoolError, ConnectTimeoutError
from pip._vendor.urllib3.exceptions import HTTPError as _HTTPError
from pip._vendor.urllib3.exceptions import InvalidHeader as _InvalidHeader
from pip._vendor.urllib3.exceptions import (
    LocationValueError,
    MaxRetryError,
    NewConnectionError,
    ProtocolError,
