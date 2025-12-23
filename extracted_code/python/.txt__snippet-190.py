import sys
import subprocess
from pathlib import Path
from . import InstallPaths, detect_platform
def run(cmd, cwd=None, check=True):
