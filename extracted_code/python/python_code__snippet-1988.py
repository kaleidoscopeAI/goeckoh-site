def cert_verify(
    self,
    conn: ConnectionPool,
    url: str,
    verify: Union[bool, str],
    cert: Optional[Union[str, Tuple[str, str]]],
) -> None:
    super().cert_verify(conn=conn, url=url, verify=False, cert=cert)


