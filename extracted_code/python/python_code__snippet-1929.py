def __init__(self, content_type: str, request_desc: str) -> None:
    super().__init__(content_type, request_desc)
    self.content_type = content_type
    self.request_desc = request_desc


