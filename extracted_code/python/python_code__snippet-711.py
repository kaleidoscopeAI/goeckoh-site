import logging
import sys
from collections import defaultdict
from itertools import chain
from typing import DefaultDict, Iterable, List, Optional, Set, Tuple

from pip._vendor.packaging import specifiers
from pip._vendor.packaging.requirements import Requirement

from pip._internal.cache import WheelCache
from pip._internal.exceptions import (
    BestVersionAlreadyInstalled,
    DistributionNotFound,
    HashError,
    HashErrors,
    InstallationError,
    NoneMetadataError,
    UnsupportedPythonVersion,
