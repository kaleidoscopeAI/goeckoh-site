"""
A logging Filter that excludes records from a logger (or its children).
"""

def filter(self, record: logging.LogRecord) -> bool:
    # The base Filter class allows only records from a logger (or its
    # children).
    return not super().filter(record)


