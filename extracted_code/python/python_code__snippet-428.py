from pip._vendor.urllib3.fields import RequestField
from pip._vendor.urllib3.filepost import encode_multipart_formdata
from pip._vendor.urllib3.util import parse_url

from ._internal_utils import to_native_string, unicode_is_ascii
from .auth import HTTPBasicAuth
from .compat import (
    Callable,
    JSONDecodeError,
    Mapping,
    basestring,
    builtin_str,
    chardet,
    cookielib,
