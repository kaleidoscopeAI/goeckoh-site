class SafeTransport(xmlrpclib.SafeTransport):

    def __init__(self, timeout, use_datetime=0):
        self.timeout = timeout
        xmlrpclib.SafeTransport.__init__(self, use_datetime)

    def make_connection(self, host):
        h, eh, kwargs = self.get_host_info(host)
        if not kwargs:
            kwargs = {}
        kwargs['timeout'] = self.timeout
        if not self._connection or host != self._connection[0]:
            self._extra_headers = eh
            self._connection = host, httplib.HTTPSConnection(
                h, None, **kwargs)
        return self._connection[1]


