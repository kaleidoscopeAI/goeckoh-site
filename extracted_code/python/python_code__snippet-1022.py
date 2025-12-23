import configparser
import logging
import os
from typing import List, Optional, Tuple

from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import HiddenText, display_path
from pip._internal.utils.subprocess import make_command
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs.versioncontrol import (
    RevOptions,
    VersionControl,
    find_path_to_project_root_from_repo_root,
    vcs,
