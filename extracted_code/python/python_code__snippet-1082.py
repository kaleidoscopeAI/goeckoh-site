from __future__ import annotations

import hashlib
import os
from textwrap import dedent
from typing import IO, TYPE_CHECKING

from pip._vendor.cachecontrol.cache import BaseCache, SeparateBodyBaseCache
from pip._vendor.cachecontrol.controller import CacheController

