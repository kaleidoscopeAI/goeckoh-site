import logging
from typing import Optional, Sequence

from pip._internal.build_env import BuildEnvironment
from pip._internal.utils.logging import indent_log
from pip._internal.utils.setuptools_build import make_setuptools_develop_args
from pip._internal.utils.subprocess import call_subprocess

