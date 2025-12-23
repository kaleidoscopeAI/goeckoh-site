    import logging

    from pip._vendor.tenacity import RetryCallState


def before_sleep_nothing(retry_state: "RetryCallState") -> None:
    """Before call strategy that does nothing."""


def before_sleep_log(
    logger: "logging.Logger",
    log_level: int,
    exc_info: bool = False,
