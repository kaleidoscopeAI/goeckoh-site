import contextlib
import itertools
import logging
import sys
import time
from typing import IO, Generator, Optional

from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.logging import get_indentation

