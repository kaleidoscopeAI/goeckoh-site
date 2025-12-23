"""HTTP connection error"""

def __init__(
    self,
    error_msg: str,
    response: Optional[Response] = None,
    request: Optional[Request] = None,
) -> None:
    """
    Initialize NetworkConnectionError with  `request` and `response`
    objects.
    """
    self.response = response
    self.request = request
    self.error_msg = error_msg
    if (
        self.response is not None
        and not self.request
        and hasattr(response, "request")
    ):
        self.request = self.response.request
    super().__init__(error_msg, response, request)

def __str__(self) -> str:
    return str(self.error_msg)


