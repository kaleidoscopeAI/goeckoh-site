def get_recent(table: str = "facts", limit: int = 20):
    return brain.mem.recent(table, limit)

