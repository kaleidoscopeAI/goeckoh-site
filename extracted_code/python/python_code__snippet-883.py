import logging
from email.message import Message
from email.parser import Parser
from typing import Tuple
from zipfile import BadZipFile, ZipFile

from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.exceptions import UnsupportedWheel

