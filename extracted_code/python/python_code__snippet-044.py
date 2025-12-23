from __future__ import absolute_import

import collections
import functools
import logging

from ._collections import HTTPHeaderDict, RecentlyUsedContainer
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool, port_by_scheme
from .exceptions import (
    LocationValueError,
    MaxRetryError,
    ProxySchemeUnknown,
    ProxySchemeUnsupported,
    URLSchemeUnknown,
