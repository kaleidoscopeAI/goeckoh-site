import re
import sys
import typing

from .util import (
    col,
    line,
    lineno,
    _collapse_string_to_ranges,
    replaced_by_pep8,
