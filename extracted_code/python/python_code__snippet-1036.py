import logging
import os
import re
from typing import List, Optional, Tuple

from pip._internal.utils.misc import (
    HiddenText,
    display_path,
    is_console_interactive,
    is_installable_dir,
    split_auth_from_netloc,
