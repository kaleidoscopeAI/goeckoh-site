from pip._vendor.urllib3.exceptions import ProxyError as _ProxyError
from pip._vendor.urllib3.exceptions import ReadTimeoutError, ResponseError
from pip._vendor.urllib3.exceptions import SSLError as _SSLError
from pip._vendor.urllib3.poolmanager import PoolManager, proxy_from_url
from pip._vendor.urllib3.util import Timeout as TimeoutSauce
from pip._vendor.urllib3.util import parse_url
from pip._vendor.urllib3.util.retry import Retry

from .auth import _basic_auth_str
from .compat import basestring, urlparse
from .cookies import extract_cookies_to_jar
from .exceptions import (
    ConnectionError,
    ConnectTimeout,
    InvalidHeader,
    InvalidProxyURL,
    InvalidSchema,
    InvalidURL,
    ProxyError,
    ReadTimeout,
    RetryError,
    SSLError,
