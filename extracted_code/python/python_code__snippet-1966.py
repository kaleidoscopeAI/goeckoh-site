def __init__(self, level: int) -> None:
    self.level = level

def filter(self, record: logging.LogRecord) -> bool:
    return record.levelno < self.level


