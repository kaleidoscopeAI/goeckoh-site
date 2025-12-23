def __init__(
    self,
    link: Link,
    persistent: bool,
):
    self.link = link
    self.persistent = persistent
    self.origin: Optional[DirectUrl] = None
    origin_direct_url_path = Path(self.link.file_path).parent / ORIGIN_JSON_NAME
    if origin_direct_url_path.exists():
        try:
            self.origin = DirectUrl.from_json(
                origin_direct_url_path.read_text(encoding="utf-8")
            )
        except Exception as e:
            logger.warning(
                "Ignoring invalid cache entry origin file %s for %s (%s)",
                origin_direct_url_path,
                link.filename,
                e,
            )


