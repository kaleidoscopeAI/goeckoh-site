import argparse
import sys
from typing import Iterable, List, Optional

from .. import __version__
from ..universaldetector import UniversalDetector


def description_of(
    lines: Iterable[bytes],
    name: str = "stdin",
    minimal: bool = False,
    should_rename_legacy: bool = False,
