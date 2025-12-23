import functools
import logging
import re
from typing import NewType, Optional, Tuple, cast

from pip._vendor.packaging import specifiers, version
from pip._vendor.packaging.requirements import Requirement

