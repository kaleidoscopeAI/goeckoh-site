import logging
from optparse import Values
from typing import List

from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.cli.req_command import SessionCommandMixin, warn_if_run_as_root
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.exceptions import InstallationError
from pip._internal.req import parse_requirements
from pip._internal.req.constructors import (
    install_req_from_line,
    install_req_from_parsed_requirement,
