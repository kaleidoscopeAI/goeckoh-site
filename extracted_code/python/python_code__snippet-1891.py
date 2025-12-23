def make(
    self, specification: str, options: Optional[Dict[str, Any]] = None
) -> List[str]:
    _raise_for_invalid_entrypoint(specification)
    return super().make(specification, options)


