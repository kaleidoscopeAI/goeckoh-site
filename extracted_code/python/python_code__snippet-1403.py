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


