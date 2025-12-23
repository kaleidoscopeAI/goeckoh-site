def decorator(func):
    def wrapper(*args, **kwargs):
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                print(f"[SELF-CORRECT] Error in {func.__name__}: {e}. Retrying {attempt + 1}/{max_retries}...")
                time.sleep(delay)
        raise RuntimeError(f"Failed after {max_retries} retries in {func.__name__}") from last_exc

    return wrapper

return decorator


