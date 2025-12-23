  def __init__(self):
      self.mem=Memory(DB_PATH); self.bus=Broadcaster()
      self.state=BrainState()
      self._rng=np.random.RandomState(101)
      self.avatar = AvatarSynth(OUT_AVATAR)
      self._seen_files: set[str] = set()

  # --- headless ingest (no user needed) ---
  def _poll_inbox(self) -> int:
      added = 0
      for p in sorted(INBOX.glob("*")):
          if not p.is_file(): continue
          if p.suffix.lower() not in {".txt",".md",".html",".htm"}: continue
          fp = str(p.resolve())
          if fp in self._seen_files: continue
          try:
               txt = p.read_text(encoding="utf-8", errors="ignore")
               lang = detect(txt[:500]) if txt.strip() else "en"
               self.mem.teach(txt, lang)
               self.mem.log(self.state.tick, "aut_ingest", {"file":p.name,"bytes":len(txt)})
               self._seen_files.add(fp); added += 1
          except Exception as e:
               self.mem.log(self.state.tick, "aut_ingest_err", {"file":p.name,"err":str(e)})
      return added

  # --- one anneal+energy step; returns energetics dict ---
  def _anneal(self) -> Dict[str,float]:
      E, ids = self.mem.embeddings(max_items=192)
      if E.size==0:
          return {"H_bits":0.0,"S_field":0.0,"L":0.0}
      N=E.shape[0]
      edges = ring_edges(N, k=max(4,min(12,N-1)))
      S=np.zeros(N)
      for i in range(N):
          var=mc_var(E,i,k=min(8,N-1), sigma=self.state.sigma, M=4, rng=self._rng)
          S[i]=stability(var)
      en=energetics(E,S,edges,self.state.sigma)
      self.mem.log(self.state.tick, "energetics", en)
      self.state.anneal_step += 1
      self.state.sigma = anneal_sigma(SIGMA0, GAMMA, self.state.anneal_step, SIGMA_MIN)
      return en

  async def loop(self):
      while True:
          try:
               self.state.tick += 1
               # autonomous ingest
               if self.state.tick % AUTON_INGEST_EVERY == 0:
                   n = self._poll_inbox()
                   if n: await self.bus.pub({"type":"ingest","data":{"tick":self.state.tick,"files":n}})
               # periodic anneal & avatar render
               if self.state.tick % REFLECT_EVERY == 0:
                   en = self._anneal()
                   await self.bus.pub({"type":"energetics","data":{"tick":self.state.tick, **en, "sigma": self.state.sigma}})
                   # sonify (short tone) → optional; informs color/tempo implicitly
                   maps=default_maps(en["H_bits"], en["S_field"], latency=0.2, fitness=max(0.0, 1.0-en["H_bits"]))
                   sig=synth_signal(0.8, 22050, maps["a"], maps["m"], maps["rho"], maps["fc"])
                   write_wav_mono16(OUT_AUDIO/f"onbrain_{self.state.tick}.wav", 22050, sig)

