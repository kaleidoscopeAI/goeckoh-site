from __future__ import absolute_import

import io
import logging
import sys
import warnings
import zlib
from contextlib import contextmanager
from socket import error as SocketError
from socket import timeout as SocketTimeout

