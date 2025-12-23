timeout: Optional[int] = None

def __init__(
    self,
    *args: Any,
    retries: int = 0,
    cache: Optional[str] = None,
    trusted_hosts: Sequence[str] = (),
    index_urls: Optional[List[str]] = None,
    ssl_context: Optional["SSLContext"] = None,
    **kwargs: Any,
) -> None:
    """
    :param trusted_hosts: Domains not to emit warnings for when not using
        HTTPS.
    """
    super().__init__(*args, **kwargs)

    # Namespace the attribute with "pip_" just in case to prevent
    # possible conflicts with the base class.
    self.pip_trusted_origins: List[Tuple[str, Optional[int]]] = []

    # Attach our User Agent to the request
    self.headers["User-Agent"] = user_agent()

    # Attach our Authentication handler to the session
    self.auth = MultiDomainBasicAuth(index_urls=index_urls)

    # Create our urllib3.Retry instance which will allow us to customize
    # how we handle retries.
    retries = urllib3.Retry(
        # Set the total number of retries that a particular request can
        # have.
        total=retries,
        # A 503 error from PyPI typically means that the Fastly -> Origin
        # connection got interrupted in some way. A 503 error in general
        # is typically considered a transient error so we'll go ahead and
        # retry it.
        # A 500 may indicate transient error in Amazon S3
        # A 502 may be a transient error from a CDN like CloudFlare or CloudFront
        # A 520 or 527 - may indicate transient error in CloudFlare
        status_forcelist=[500, 502, 503, 520, 527],
        # Add a small amount of back off between failed requests in
        # order to prevent hammering the service.
        backoff_factor=0.25,
    )  # type: ignore

    # Our Insecure HTTPAdapter disables HTTPS validation. It does not
    # support caching so we'll use it for all http:// URLs.
    # If caching is disabled, we will also use it for
    # https:// hosts that we've marked as ignoring
    # TLS errors for (trusted-hosts).
    insecure_adapter = InsecureHTTPAdapter(max_retries=retries)

    # We want to _only_ cache responses on securely fetched origins or when
    # the host is specified as trusted. We do this because
    # we can't validate the response of an insecurely/untrusted fetched
    # origin, and we don't want someone to be able to poison the cache and
    # require manual eviction from the cache to fix it.
    if cache:
        secure_adapter = CacheControlAdapter(
            cache=SafeFileCache(cache),
            max_retries=retries,
            ssl_context=ssl_context,
        )
        self._trusted_host_adapter = InsecureCacheControlAdapter(
            cache=SafeFileCache(cache),
            max_retries=retries,
        )
    else:
        secure_adapter = HTTPAdapter(max_retries=retries, ssl_context=ssl_context)
        self._trusted_host_adapter = insecure_adapter

    self.mount("https://", secure_adapter)
    self.mount("http://", insecure_adapter)

    # Enable file:// urls
    self.mount("file://", LocalFSAdapter())

    for host in trusted_hosts:
        self.add_trusted_host(host, suppress_logging=True)

def update_index_urls(self, new_index_urls: List[str]) -> None:
    """
    :param new_index_urls: New index urls to update the authentication
        handler with.
    """
    self.auth.index_urls = new_index_urls

def add_trusted_host(
    self, host: str, source: Optional[str] = None, suppress_logging: bool = False
) -> None:
    """
    :param host: It is okay to provide a host that has previously been
        added.
    :param source: An optional source string, for logging where the host
        string came from.
    """
    if not suppress_logging:
        msg = f"adding trusted host: {host!r}"
        if source is not None:
            msg += f" (from {source})"
        logger.info(msg)

    parsed_host, parsed_port = parse_netloc(host)
    if parsed_host is None:
        raise ValueError(f"Trusted host URL must include a host part: {host!r}")
    if (parsed_host, parsed_port) not in self.pip_trusted_origins:
        self.pip_trusted_origins.append((parsed_host, parsed_port))

    self.mount(
        build_url_from_netloc(host, scheme="http") + "/", self._trusted_host_adapter
    )
    self.mount(build_url_from_netloc(host) + "/", self._trusted_host_adapter)
    if not parsed_port:
        self.mount(
            build_url_from_netloc(host, scheme="http") + ":",
            self._trusted_host_adapter,
        )
        # Mount wildcard ports for the same host.
        self.mount(build_url_from_netloc(host) + ":", self._trusted_host_adapter)

def iter_secure_origins(self) -> Generator[SecureOrigin, None, None]:
    yield from SECURE_ORIGINS
    for host, port in self.pip_trusted_origins:
        yield ("*", host, "*" if port is None else port)

def is_secure_origin(self, location: Link) -> bool:
    # Determine if this url used a secure transport mechanism
    parsed = urllib.parse.urlparse(str(location))
    origin_protocol, origin_host, origin_port = (
        parsed.scheme,
        parsed.hostname,
        parsed.port,
    )

    # The protocol to use to see if the protocol matches.
    # Don't count the repository type as part of the protocol: in
    # cases such as "git+ssh", only use "ssh". (I.e., Only verify against
    # the last scheme.)
    origin_protocol = origin_protocol.rsplit("+", 1)[-1]

    # Determine if our origin is a secure origin by looking through our
    # hardcoded list of secure origins, as well as any additional ones
    # configured on this PackageFinder instance.
    for secure_origin in self.iter_secure_origins():
        secure_protocol, secure_host, secure_port = secure_origin
        if origin_protocol != secure_protocol and secure_protocol != "*":
            continue

        try:
            addr = ipaddress.ip_address(origin_host or "")
            network = ipaddress.ip_network(secure_host)
        except ValueError:
            # We don't have both a valid address or a valid network, so
            # we'll check this origin against hostnames.
            if (
                origin_host
                and origin_host.lower() != secure_host.lower()
                and secure_host != "*"
            ):
                continue
        else:
            # We have a valid address and network, so see if the address
            # is contained within the network.
            if addr not in network:
                continue

        # Check to see if the port matches.
        if (
            origin_port != secure_port
            and secure_port != "*"
            and secure_port is not None
        ):
            continue

        # If we've gotten here, then this origin matches the current
        # secure origin and we should return True
        return True

    # If we've gotten to this point, then the origin isn't secure and we
    # will not accept it as a valid location to search. We will however
    # log a warning that we are ignoring it.
    logger.warning(
        "The repository located at %s is not a trusted or secure host and "
        "is being ignored. If this repository is available via HTTPS we "
        "recommend you use HTTPS instead, otherwise you may silence "
        "this warning and allow it anyway with '--trusted-host %s'.",
        origin_host,
        origin_host,
    )

    return False

def request(self, method: str, url: str, *args: Any, **kwargs: Any) -> Response:
    # Allow setting a default timeout on a session
    kwargs.setdefault("timeout", self.timeout)
    # Allow setting a default proxies on a session
    kwargs.setdefault("proxies", self.proxies)

    # Dispatch the actual request
    return super().request(method, url, *args, **kwargs)


