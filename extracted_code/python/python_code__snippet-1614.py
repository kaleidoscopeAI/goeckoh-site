import functools
import logging
import logging.config
import optparse
import os
import sys
import traceback
from optparse import Values
from typing import Any, Callable, List, Optional, Tuple

from pip._vendor.rich import traceback as rich_traceback

from pip._internal.cli import cmdoptions
from pip._internal.cli.command_context import CommandContextMixIn
from pip._internal.cli.parser import ConfigOptionParser, UpdatingDefaultsHelpFormatter
from pip._internal.cli.status_codes import (
    ERROR,
    PREVIOUS_BUILD_DIR_ERROR,
    UNKNOWN_ERROR,
    VIRTUALENV_NOT_FOUND,
