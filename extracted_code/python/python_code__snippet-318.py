from contextlib import contextmanager
from io import StringIO
import sys
import os


class StreamTTY(StringIO):
    def isatty(self):
        return True

class StreamNonTTY(StringIO):
    def isatty(self):
        return False

