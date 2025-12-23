import warnings

from ..exceptions import DependencyWarning

warnings.warn(
    (
        "SOCKS support in urllib3 requires the installation of optional "
        "dependencies: specifically, PySocks.  For more information, see "
        "https://urllib3.readthedocs.io/en/1.26.x/contrib.html#socks-proxies"
    ),
    DependencyWarning,
)
raise

