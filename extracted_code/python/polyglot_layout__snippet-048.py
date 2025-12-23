     for u, v, data in self.G.edges(data=True):
       b: Bond = data['bond']
       pu = self.G.nodes[u]['node'].pos
       pv = self.G.nodes[v]['node'].pos
       L = float(np.linalg.norm(pv - pu))
       tension += abs(L - b.rest)
       energy += 0.5 * b.k * (L - b.rest)**2
     m = max(1, self.G.number_of_edges())
     return {"tension": tension / m, "energy": energy / m, "size": self.G.number_of_nodes()}

  def apply_adjustments(self, adj: Dict[str, float]):
    ks = float(adj.get("k_scale", 1.0))
    rs = float(adj.get("rest_scale", 1.0))
    ks = max(0.25, min(ks, 4.0))
    rs = max(0.5, min(rs, 1.5))
    for _, _, data in self.G.edges(data=True):
       b: Bond = data['bond']
       b.k *= ks
       b.rest *= rs

# Reflection and Adjustment
def make_reflection(tick: int, m: Dict[str, float]) -> str:
  t = m.get("tension", 0.0)
  e = m.get("energy", 0.0)
  n = m.get("size", 0)
  mood = "calm" if t < 0.01 else "strained" if t < 0.04 else "overstretched"
  return f"[tick={tick}] Tension={t:.5f} Energy={e:.5f} Size={n}. State feels {mood}. Strategy: {'tighten springs' if t < 0.02 else 'loosen a bit' if t > 0.05
else 'hold steady'}."

def heuristic_adjust(m: Dict[str, float]) -> Dict[str, float]:
  t = m.get("tension", 0.0)
  if t < 0.015:
     return {"k_scale": 1.10, "rest_scale": 0.98}
  if t > 0.050:
     return {"k_scale": 0.95, "rest_scale": 1.03}
  return {"k_scale": 1.00, "rest_scale": 1.00}

def ask_ollama_refine(metrics: Dict[str, float], reflection: str) -> Dict[str, Any]:
  sys_p = "You optimize a spring-mesh cube. Reply STRICT JSON only: {\"k_scale\":float, \"rest_scale\":float}."
  user_p = f"Metrics: {metrics}\nReflection: {reflection}\nReturn only JSON."
  try:
     r = requests.post(f"{OLLAMA_URL}/api/chat", json={"model": OLLAMA_MODEL, "messages": [{"role": "system", "content": sys_p}, {"role": "user",
"content": user_p}], "stream": False}, timeout=15)
     r.raise_for_status()
     content = r.json().get("message", {}).get("content", "").strip()
     data = json.loads(content)
     if not all(k in data for k in ("k_scale", "rest_scale")):
        raise ValueError("Missing keys")
     return {"ok": True, "adjust": data, "raw": content}
  except Exception as e:
     return {"ok": False, "adjust": heuristic_adjust(metrics), "error": str(e)}

# Crawler and Autonomous Ingestion
def fetch_url(url: str) -> Tuple[str, str]:
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

def x_search(query: str, limit: int = 5) -> List[str]:
  # Simulate X search tool; in production, integrate with actual API or tool.
  # For demo, hardcode or use requests to X API if available.
  # Here, placeholder returns dummy URLs.
  return [f"https://example.com/post/{i}" for i in range(limit)]

# Orchestrator

