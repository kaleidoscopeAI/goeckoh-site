import os
import re
import sys
from typing import List, Optional

from pip._internal.locations import site_packages, user_site
from pip._internal.utils.virtualenv import (
    running_under_virtualenv,
    virtualenv_no_global,
