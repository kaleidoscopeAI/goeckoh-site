from typing import Optional

from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.distributions.base import AbstractDistribution
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import (
    BaseDistribution,
    FilesystemWheel,
    get_wheel_distribution,
