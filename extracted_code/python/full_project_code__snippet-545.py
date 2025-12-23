class SSLContext(object):  # Platform-specific: Python 2
    def __init__(self, protocol_version):
        self.protocol = protocol_version
        # Use default values from a real SSLContext
        self.check_hostname = False
        self.verify_mode = ssl.CERT_NONE
        self.ca_certs = None
        self.options = 0
        self.certfile = None
        self.keyfile = None
        self.ciphers = None

    def load_cert_chain(self, certfile, keyfile):
        self.certfile = certfile
        self.keyfile = keyfile

    def load_verify_locations(self, cafile=None, capath=None, cadata=None):
        self.ca_certs = cafile

        if capath is not None:
            raise SSLError("CA directories not supported in older Pythons")

        if cadata is not None:
            raise SSLError("CA data not supported in older Pythons")

    def set_ciphers(self, cipher_suite):
        self.ciphers = cipher_suite

    def wrap_socket(self, socket, server_hostname=None, server_side=False):
        warnings.warn(
            "A true SSLContext object is not available. This prevents "
            "urllib3 from configuring SSL appropriately and may cause "
            "certain SSL connections to fail. You can upgrade to a newer "
            "version of Python to solve this. For more information, see "
            "https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html"
            "#ssl-warnings",
            InsecurePlatformWarning,
        )
        kwargs = {
            "keyfile": self.keyfile,
            "certfile": self.certfile,
            "ca_certs": self.ca_certs,
            "cert_reqs": self.verify_mode,
            "ssl_version": self.protocol,
            "server_side": server_side,
        }
        return wrap_socket(socket, ciphers=self.ciphers, **kwargs)


