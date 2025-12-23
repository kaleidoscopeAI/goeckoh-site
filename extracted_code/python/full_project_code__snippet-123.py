from .util import SKIP_HEADER, SKIPPABLE_HEADERS, connection
from .util.ssl_ import (
    assert_fingerprint,
    create_urllib3_context,
    is_ipaddress,
    resolve_cert_reqs,
    resolve_ssl_version,
    ssl_wrap_socket,
