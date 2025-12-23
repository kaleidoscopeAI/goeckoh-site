import shutil
import subprocess
import sys
from pathlib import Path
from . import InstallPaths, detect_platform
def run(cmd, cwd=None, check=True):
