    import logging

    from pip._vendor.tenacity import RetryCallState


def after_nothing(retry_state: "RetryCallState") -> None:
    """After call strategy that does nothing."""


def after_log(
    logger: "logging.Logger",
    log_level: int,
    sec_format: str = "%0.3f",
