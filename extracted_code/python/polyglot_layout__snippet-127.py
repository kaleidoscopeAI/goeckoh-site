  if url:
      try:
          r = requests.get(url, timeout=8, headers={"User-Agent":"SeedCrystal/1.0"}); r.raise_for_status()
          soup = BeautifulSoup(r.text, "html.parser")
          for tag in soup(["script","style","noscript"]): tag.decompose()
          title = (soup.title.text.strip() if soup.title else url)[:200]
          import re
          text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:10000]
      except Exception as e:
          return {"ok": False, "error": str(e)}
      doc_id = orch.mem.add_doc_with_embed(url, title, text)
  else:
      doc_id = orch._ingest_local()
  return {"ok": True, "doc_id": doc_id}

