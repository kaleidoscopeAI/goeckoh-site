from __future__ import absolute_import

import email
import logging
import re
import time
import warnings
from collections import namedtuple
from itertools import takewhile

from ..exceptions import (
    ConnectTimeoutError,
    InvalidHeader,
    MaxRetryError,
    ProtocolError,
    ProxyError,
    ReadTimeoutError,
    ResponseError,
