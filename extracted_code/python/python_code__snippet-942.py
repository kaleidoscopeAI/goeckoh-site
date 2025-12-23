import logging
import os
import shutil
from optparse import Values
from typing import List

from pip._internal.cache import WheelCache
from pip._internal.cli import cmdoptions
from pip._internal.cli.req_command import RequirementCommand, with_cleanup
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.exceptions import CommandError
from pip._internal.operations.build.build_tracker import get_build_tracker
from pip._internal.req.req_install import (
    InstallRequirement,
    check_legacy_setup_py_options,
