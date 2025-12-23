"""
A class mixin for command classes needing _build_session().
"""

def __init__(self) -> None:
    super().__init__()
    self._session: Optional[PipSession] = None

@classmethod
def _get_index_urls(cls, options: Values) -> Optional[List[str]]:
    """Return a list of index urls from user-provided options."""
    index_urls = []
    if not getattr(options, "no_index", False):
        url = getattr(options, "index_url", None)
        if url:
            index_urls.append(url)
    urls = getattr(options, "extra_index_urls", None)
    if urls:
        index_urls.extend(urls)
    # Return None rather than an empty list
    return index_urls or None

def get_default_session(self, options: Values) -> PipSession:
    """Get a default-managed session."""
    if self._session is None:
        self._session = self.enter_context(self._build_session(options))
        # there's no type annotation on requests.Session, so it's
        # automatically ContextManager[Any] and self._session becomes Any,
        # then https://github.com/python/mypy/issues/7696 kicks in
        assert self._session is not None
    return self._session

def _build_session(
    self,
    options: Values,
    retries: Optional[int] = None,
    timeout: Optional[int] = None,
    fallback_to_certifi: bool = False,
) -> PipSession:
    cache_dir = options.cache_dir
    assert not cache_dir or os.path.isabs(cache_dir)

    if "truststore" in options.features_enabled:
        try:
            ssl_context = _create_truststore_ssl_context()
        except Exception:
            if not fallback_to_certifi:
                raise
            ssl_context = None
    else:
        ssl_context = None

    session = PipSession(
        cache=os.path.join(cache_dir, "http-v2") if cache_dir else None,
        retries=retries if retries is not None else options.retries,
        trusted_hosts=options.trusted_hosts,
        index_urls=self._get_index_urls(options),
        ssl_context=ssl_context,
    )

    # Handle custom ca-bundles from the user
    if options.cert:
        session.verify = options.cert

    # Handle SSL client certificate
    if options.client_cert:
        session.cert = options.client_cert

    # Handle timeouts
    if options.timeout or timeout:
        session.timeout = timeout if timeout is not None else options.timeout

    # Handle configured proxies
    if options.proxy:
        session.proxies = {
            "http": options.proxy,
            "https": options.proxy,
        }

    # Determine if we can prompt the user for authentication or not
    session.auth.prompting = not options.no_input
    session.auth.keyring_provider = options.keyring_provider

    return session


