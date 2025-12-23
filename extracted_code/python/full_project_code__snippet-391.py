import codecs
import contextlib
import io
import os
import re
import socket
import struct
import sys
import tempfile
import warnings
import zipfile
from collections import OrderedDict

from pip._vendor.urllib3.util import make_headers, parse_url

from . import certs
from .__version__ import __version__

