    from ..packages.backports.makefile import backport_makefile

import logging
import ssl
import sys
import warnings

from .. import util
from ..packages import six
from ..util.ssl_ import PROTOCOL_TLS_CLIENT

