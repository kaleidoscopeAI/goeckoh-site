import re
import string
import urllib.parse
from typing import List, Optional as TOptional, Set

from pip._vendor.pyparsing import (  # noqa
    Combine,
    Literal as L,
    Optional,
    ParseException,
    Regex,
    Word,
    ZeroOrMore,
    originalTextFor,
    stringEnd,
    stringStart,
