import operator
import os
import platform
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pip._vendor.pyparsing import (  # noqa: N817
    Forward,
    Group,
    Literal as L,
    ParseException,
    ParseResults,
    QuotedString,
    ZeroOrMore,
    stringEnd,
    stringStart,
