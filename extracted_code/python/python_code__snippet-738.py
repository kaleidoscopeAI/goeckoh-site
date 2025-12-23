import collections
import logging
import os
from typing import Container, Dict, Generator, Iterable, List, NamedTuple, Optional, Set

from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import Version

from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.metadata import BaseDistribution, get_environment
from pip._internal.req.constructors import (
    install_req_from_editable,
    install_req_from_line,
