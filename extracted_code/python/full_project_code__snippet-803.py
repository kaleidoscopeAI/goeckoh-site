"""Retrying controller."""

def __call__(
    self,
    fn: t.Callable[..., WrappedFnReturnT],
    *args: t.Any,
    **kwargs: t.Any,
) -> WrappedFnReturnT:
    self.begin()

    retry_state = RetryCallState(retry_object=self, fn=fn, args=args, kwargs=kwargs)
    while True:
        do = self.iter(retry_state=retry_state)
        if isinstance(do, DoAttempt):
            try:
                result = fn(*args, **kwargs)
            except BaseException:  # noqa: B902
                retry_state.set_exception(sys.exc_info())  # type: ignore[arg-type]
            else:
                retry_state.set_result(result)
        elif isinstance(do, DoSleep):
            retry_state.prepare_for_next_attempt()
            self.sleep(do)
        else:
            return do  # type: ignore[no-any-return]


