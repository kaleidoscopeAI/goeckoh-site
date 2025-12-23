from .ssl_ import create_urllib3_context, resolve_cert_reqs, resolve_ssl_version


def connection_requires_http_tunnel(
    proxy_url=None, proxy_config=None, destination_scheme=None
