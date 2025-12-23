from __future__ import absolute_import

import errno
import logging
import re
import socket
import sys
import warnings
from socket import error as SocketError
from socket import timeout as SocketTimeout

from ._collections import HTTPHeaderDict
from .connection import (
    BaseSSLError,
    BrokenPipeError,
    DummyConnection,
    HTTPConnection,
    HTTPException,
    HTTPSConnection,
    VerifiedHTTPSConnection,
    port_by_scheme,
