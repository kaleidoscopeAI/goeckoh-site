import html.entities
import re
import sys
import typing

from . import __diag__
from .core import *
from .util import (
    _bslash,
    _flatten,
    _escape_regex_range_chars,
    replaced_by_pep8,
