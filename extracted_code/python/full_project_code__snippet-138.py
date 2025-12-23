from __future__ import absolute_import

import warnings
from logging import getLogger

from ntlm import ntlm

from .. import HTTPSConnectionPool
from ..packages.six.moves.http_client import HTTPSConnection

