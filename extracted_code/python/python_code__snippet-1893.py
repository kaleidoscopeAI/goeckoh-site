def __init__(self, path: str, content_type: Optional[str]) -> None:
    self.path = path
    if content_type is None:
        self.content_type = mimetypes.guess_type(path)[0]
    else:
        self.content_type = content_type


