import importlib.util
import os
from collections import namedtuple
from typing import Any, List, Optional

from pip._vendor import tomli
from pip._vendor.packaging.requirements import InvalidRequirement, Requirement

from pip._internal.exceptions import (
    InstallationError,
    InvalidPyProjectBuildRequires,
    MissingPyProjectBuildRequires,
