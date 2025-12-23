    class RLock:
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_value, traceback):
            pass


from collections import OrderedDict

from .exceptions import InvalidHeader
from .packages import six
from .packages.six import iterkeys, itervalues

