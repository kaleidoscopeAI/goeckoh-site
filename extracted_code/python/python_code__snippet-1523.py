class DirectUrlValidationError(Exception):
    pass


def _get(
    d: Dict[str, Any], expected_type: Type[T], key: str, default: Optional[T] = None
