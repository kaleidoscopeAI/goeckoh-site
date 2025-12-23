def reveal_type(__obj: T) -> T:
    """Reveal the inferred type of a variable.

    When a static type checker encounters a call to ``reveal_type()``,
    it will emit the inferred type of the argument::

        x: int = 1
        reveal_type(x)

    Running a static type checker (e.g., ``mypy``) on this example
    will produce output similar to 'Revealed type is "builtins.int"'.

    At runtime, the function prints the runtime type of the
    argument and returns it unchanged.

    """
    print(f"Runtime type is {type(__obj).__name__!r}", file=sys.stderr)
    return __obj


