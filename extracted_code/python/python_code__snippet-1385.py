import hashlib
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional

from pip._internal.exceptions import HashMismatch, HashMissing, InstallationError
from pip._internal.utils.misc import read_chunks

