import logging
import os

from pip._internal.build_env import BuildEnvironment
from pip._internal.cli.spinners import open_spinner
from pip._internal.exceptions import (
    InstallationError,
    InstallationSubprocessError,
    MetadataGenerationFailed,
