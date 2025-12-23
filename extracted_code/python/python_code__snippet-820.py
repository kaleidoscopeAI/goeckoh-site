def _onerror_ignore(*_args: Any) -> None:
    pass


def _onerror_reraise(*_args: Any) -> None:
    raise


def rmtree_errorhandler(
    func: FunctionType,
    path: Path,
    exc_info: Union[ExcInfo, BaseException],
    *,
    onexc: OnExc = _onerror_reraise,
