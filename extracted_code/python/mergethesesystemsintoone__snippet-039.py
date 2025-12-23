def recent(table: str = Query("states"), limit: int = 50):
    return {"ok": True, "rows": orch.mem.recent(table, limit)}

