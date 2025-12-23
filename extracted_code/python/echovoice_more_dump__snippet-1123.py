def format(self, record: logging.LogRecord) -> str:
    payload = {
        'time': self.formatTime(record, self.datefmt),
        'level': record.levelname,
        'logger': record.name,
        'message': record.getMessage(),
    }
    if record.exc_info:
        payload['exc'] = self.formatException(record.exc_info)
    return json.dumps(payload)


