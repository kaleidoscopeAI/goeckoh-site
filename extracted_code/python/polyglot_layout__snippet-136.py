  def __init__(self):
      self.cube = Cube(n_per_edge=6)
      self.mem = MemoryStore(DB_PATH)
      self.tick = 0
      self.bus = Broadcaster()
      self.anneal_step = 0
      self.sigma = SIGMA0
      self.theta_gel = 0.25
      self.theta_crystal = 0.08
      self.rng = np.random.RandomState(101)
      self.hbits = 0.5; self.sfield = 0.5
      self.last_caption_text = ""
      self.avatar = Avatar18k(n=18000)

  def snapshot(self) -> Dict[str, Any]:
      m = self.cube.metrics()
      return {"tick": self.tick, **m, "sigma": self.sigma, "H_bits": self.hbits, "S_field": self.sfield}

  def _ingest_local(self) -> Optional[int]:
      files = sorted(CORPUS.glob("**/*"))
      if not files: return None
      pick = random.choice([f for f in files if f.is_file()])
      try:
          text = pick.read_text(encoding="utf-8", errors="ignore")
      except Exception:
          text = ""
      title = pick.name
      return self.mem.add_doc_with_embed(pick.as_uri(), title, text)

  def _ingest_online(self) -> Optional[int]:
      # Minimal, safe fetch (e.g., wikipedia page); only if allowed.
      url = random.choice([
          "https://en.wikipedia.org/wiki/Artificial_intelligence",
          "https://en.wikipedia.org/wiki/Information_theory",
          "https://en.wikipedia.org/wiki/Signal_processing"
      ])
      try:
          r = requests.get(url, timeout=8, headers={"User-Agent":"SeedCrystal/1.0"}); r.raise_for_status()
          soup = BeautifulSoup(r.text, "html.parser")
          for tag in soup(["script","style","noscript"]): tag.decompose()
          title = (soup.title.text.strip() if soup.title else url)[:200]
          import re
          text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[:10000]
          return self.mem.add_doc_with_embed(url, title, text)
      except Exception:
          return None

  def _anneal_and_process(self):
      E, ids = self.mem.get_embeddings(max_items=128)
      if E.size == 0: return None
      N = E.shape[0]
      edges = simple_edges(N, k=max(4, min(12, N - 1)))
      S = np.zeros(N, dtype=np.float64)
      for i in range(N):
          var = monte_carlo_variance(E, i, k=min(8, N - 1), sigma=self.sigma, M=4, rng=self.rng)
          S[i] = stability_score(var)
      en = energetics(E, S, edges, self.sigma)
      self.mem.add_energetics(self.tick, self.sigma, en["H_bits"], en["S_field"], en["L"])
      self.hbits = float(en["H_bits"]); self.sfield = float(en["S_field"])

      # Sonify → STFT → Attention → Captions
      maps = default_maps(H_bits=self.hbits, S_field=self.sfield, latency=0.2, fitness=max(0.0, 1.0 - self.hbits))
      sig = synth_signal(seconds=1.6, sr=22050, a_fn=maps["a"], m_fn=maps["m"], rho_fn=maps["rho"], fc_fn=maps["fc"], alpha=0.8, beta=0.4)
      wav_path = OUT_AUDIO / f"sonification_{self.tick}.wav"
      write_wav_mono16(wav_path, 22050, sig)

      X = stft_mag(np.asarray(sig, dtype=np.float64), sr=22050, win=1024, hop=256)
      bands = make_bands(X.shape[0], H=4)
      V = head_features(X, bands)
      shapes = project_and_attention(V, E_mem=E, d=24, sigma_temp=max(self.sigma, SIGMA_MIN))

