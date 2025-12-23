import re
from typing import List, Optional, Tuple

from pip._vendor.packaging.tags import (
    PythonVersion,
    Tag,
    compatible_tags,
    cpython_tags,
    generic_tags,
    interpreter_name,
    interpreter_version,
    mac_platforms,
