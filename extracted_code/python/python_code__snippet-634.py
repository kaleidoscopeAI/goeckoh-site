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


class ServerProxy(xmlrpclib.ServerProxy):

    def __init__(self, uri, **kwargs):
        self.timeout = timeout = kwargs.pop('timeout', None)
        # The above classes only come into play if a timeout
        # is specified
        if timeout is not None:
            # scheme = splittype(uri)  # deprecated as of Python 3.8
            scheme = urlparse(uri)[0]
            use_datetime = kwargs.get('use_datetime', 0)
            if scheme == 'https':
                tcls = SafeTransport
            else:
                tcls = Transport
            kwargs['transport'] = t = tcls(timeout, use_datetime=use_datetime)
            self.transport = t
        xmlrpclib.ServerProxy.__init__(self, uri, **kwargs)


