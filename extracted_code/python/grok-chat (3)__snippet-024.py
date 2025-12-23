def self_correcting(max_retries=3, delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error in {func.__name__}: {e}. Retrying {attempt+1}/{max_retries}...")
                    time.sleep(delay)
            raise RuntimeError(f"Failed after {max_retries} retries in {func.__name__}")
        return wrapper
    return decorator

