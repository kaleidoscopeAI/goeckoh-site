from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.filesystem import test_writable_dir
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import (
    check_externally_managed,
    ensure_dir,
    get_pip_version,
    protect_pip_from_modification_on_windows,
    write_output,
