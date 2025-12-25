import sys
import os
from pathlib import Path

def is_android():
    return 'ANDROID_ARGUMENT' in os.environ

def get_root_dir():
    if getattr(sys, 'frozen', False):
        return Path(os.path.dirname(sys.executable))
    return Path(__file__).resolve().parent.parent

def get_asset_path(rel_path: str) -> str:
    return str(get_root_dir() / "assets" / rel_path)
