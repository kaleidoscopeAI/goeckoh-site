  try:
       r = requests.get(url, timeout=30, headers={"User-Agent": "SeedCrystalAGI/1.0"})
       r.raise_for_status()
       html = r.text
       soup = BeautifulSoup(html, "html.parser")
       title = (soup.title.text.strip() if soup.title else url)[:200]
       for tag in soup(["script", "style", "noscript"]):
           tag.decompose()
       import re
       text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
       return title, text[:10000]
  except Exception as e:
       return "", str(e)

