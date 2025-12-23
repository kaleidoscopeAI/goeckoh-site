def decorator(func):
    def wrapper(*args, **kwargs):
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:  # pragma: no cover - defensive retry
                last_exc = e
                print(f"[SELF-CORRECT] {func.__name__} failed: {e} (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
        raise RuntimeError(f"{func.__name__} failed after {max_retries} retries") from last_exc

    return wrapper

return decorator


