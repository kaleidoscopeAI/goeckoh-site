from __future__ import annotations

import functools
import types
import zlib
from typing import TYPE_CHECKING, Any, Collection, Mapping

from pip._vendor.requests.adapters import HTTPAdapter

from pip._vendor.cachecontrol.cache import DictCache
from pip._vendor.cachecontrol.controller import PERMANENT_REDIRECT_STATUSES, CacheController
from pip._vendor.cachecontrol.filewrapper import CallbackFileWrapper

