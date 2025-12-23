import functools
import os
import site
import sys
import sysconfig
import typing

from pip._internal.exceptions import InstallationError
from pip._internal.utils import appdirs
from pip._internal.utils.virtualenv import running_under_virtualenv

