from __future__ import annotations

import io
from typing import IO, TYPE_CHECKING, Any, Mapping, cast

from pip._vendor import msgpack
from pip._vendor.requests.structures import CaseInsensitiveDict
from pip._vendor.urllib3 import HTTPResponse

