import mimetypes
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.distributions import make_distribution_for_install_requirement
from pip._internal.distributions.installed import InstalledDistribution
from pip._internal.exceptions import (
    DirectoryUrlHashUnsupported,
    HashMismatch,
    HashUnpinned,
    InstallationError,
    MetadataInconsistent,
    NetworkConnectionError,
    VcsHashUnsupported,
