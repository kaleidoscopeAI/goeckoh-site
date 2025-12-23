def _default_exception_debug_action(
    instring: str,
    loc: int,
    expr: "ParserElement",
    exc: Exception,
    cache_hit: bool = False,
