import logging
import urllib.parse
import xmlrpc.client
from typing import TYPE_CHECKING, Tuple

from pip._internal.exceptions import NetworkConnectionError
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status

