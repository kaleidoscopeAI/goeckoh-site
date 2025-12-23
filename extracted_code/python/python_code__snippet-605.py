import os
import re
import sys
import platform

from .compat import string_types
from .util import in_venv, parse_marker
from .version import LegacyVersion as LV

