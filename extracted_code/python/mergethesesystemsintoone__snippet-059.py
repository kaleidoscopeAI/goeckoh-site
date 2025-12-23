def ingest(url: str):
    title, text = fetch_url(url)
    doc_id = orch.mem.add_doc_with_embed(url, title, text)
    return {"ok": True, "doc_id": doc_id}

