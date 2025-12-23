    import dummy_threading as threading
import zlib

from . import DistlibException
from .compat import (urljoin, urlparse, urlunparse, url2pathname, pathname2url,
                     queue, quote, unescape, build_opener,
                     HTTPRedirectHandler as BaseRedirectHandler, text_type,
                     Request, HTTPError, URLError)
from .database import Distribution, DistributionPath, make_dist
from .metadata import Metadata, MetadataInvalidError
from .util import (cached_property, ensure_slash, split_filename, get_project_data,
                   parse_requirement, parse_name_and_version, ServerProxy,
                   normalize_name)
from .version import get_scheme, UnsupportedVersionError
from .wheel import Wheel, is_compatible

