import logging
import os
import sys
import sysconfig
import typing

from pip._internal.exceptions import InvalidSchemeCombination, UserInstallationInvalid
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.virtualenv import running_under_virtualenv

from .base import change_root, get_major_minor_version, is_osx_framework

