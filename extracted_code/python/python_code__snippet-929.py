import logging
from optparse import Values
from typing import List

from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.operations.check import (
    check_package_set,
    create_package_set_from_installed,
    warn_legacy_versions_and_specifiers,
