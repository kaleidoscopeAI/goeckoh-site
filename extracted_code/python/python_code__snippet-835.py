import logging
import os
import shutil
import stat
import tarfile
import zipfile
from typing import Iterable, List, Optional
from zipfile import ZipInfo

from pip._internal.exceptions import InstallationError
from pip._internal.utils.filetypes import (
    BZ2_EXTENSIONS,
    TAR_EXTENSIONS,
    XZ_EXTENSIONS,
    ZIP_EXTENSIONS,
