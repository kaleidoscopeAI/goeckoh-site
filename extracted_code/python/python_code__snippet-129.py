class Transport(xmlrpclib.Transport):

    def __init__(self, timeout, use_datetime=0):
        self.timeout = timeout
        xmlrpclib.Transport.__init__(self, use_datetime)

    def make_connection(self, host):
        h, eh, x509 = self.get_host_info(host)
        if not self._connection or host != self._connection[0]:
            self._extra_headers = eh
            self._connection = host, httplib.HTTPConnection(h)
        return self._connection[1]


