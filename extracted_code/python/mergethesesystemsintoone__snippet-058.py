class Broadcaster:
    def __init__(self):
        self._subs: List[asyncio.Queue] = []

    def subscribe(self):
        q = asyncio.Queue(maxsize=200)
        self._subs.append(q)
        return q

    async def publish(self, msg: Dict[str, Any]):
        for q in list(self._subs):
            try:
                await q.put(msg)
            except asyncio.QueueFull:
                pass

def simple_edges(N: int, k: int = 6) -> np.ndarray:
    edges = []
    for i in range(N):
        edges.append((i, (i + 1) % N))
        for j in range(1, k // 2 + 1):
            edges.append((i, (i + j) % N))
    if not edges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(sorted({tuple(sorted(e)) for e in edges}), dtype=np.int32)

class Orchestrator:
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

    def snapshot(self) -> Dict[str, Any]:
        m = self.cube.metrics()
        return {"tick": self.tick, **m, "sigma": self.sigma}

    async def autonomous_ingest(self):
        # Search X for relevant posts, extract links, ingest one.
        links = x_search(X_SEARCH_QUERY, limit=3)
        if links:
            url = random.choice(links)
            title, text = fetch_url(url)
            if text:
                doc_id = self.mem.add_doc_with_embed(url, title, text)
                await self.bus.publish({"type": "ingest", "data": {"url": url, "doc_id": doc_id}})

    def _anneal_and_process(self):
        E, ids = self.mem.get_embeddings(max_items=128)
        if E.size == 0:
            return None
        N = E.shape[0]
        edges = simple_edges(N, k=max(4, min(12, N - 1)))
        S = np.zeros(N, dtype=np.float64)
        for i in range(N):
            var = monte_carlo_variance(E, i, k=min(8, N - 1), sigma=self.sigma, M=4, rng=self.rng)
            S[i] = stability_score(var)
        en = energetics(E, S, edges, self.sigma)
        self.mem.add_energetics(self.tick, self.sigma, en["H_bits"], en["S_field"], en["L"])
        maps = default_maps(H_bits=en["H_bits"], S_field=en["S_field"], latency=0.2, fitness=max(0.0, 1.0 - en["H_bits"]))
        sig = synth_signal(seconds=2.0, sr=22050, a_fn=maps["a"], m_fn=maps["m"], rho_fn=maps["rho"], fc_fn=maps["fc"], alpha=0.8, beta=0.4)
        wav_path = OUT_AUDIO / f"sonification_{self.tick}.wav"
        write_wav_mono16(wav_path, 22050, sig)
        x = np.array(sig, dtype=np.float64)
        X = stft_mag(x, sr=22050, win=1024, hop=256)
        bands = make_bands(X.shape[0], H=4)
        V = head_features(X, bands)
        shapes = project_and_attention(V, E_mem=E, d=24, sigma_temp=max(self.sigma, SIGMA_MIN))
        caps = captions_from_shapes(DB_PATH, shapes, top_k=3, window=6, stride=6, hbits=en["H_bits"], sfield=en["S_field"])
        if caps["captions"]:
            last = caps["captions"][-1]
            self.mem.add_caption(self.tick, last.get("caption", ""), last.get("top_ids", []), last.get("weights", []))
        with open(OUT_SHAPES / f"shapes_{self.tick}.json", "w", encoding="utf-8") as f:
            json.dump(shapes, f, ensure_ascii=False, indent=2)
        with open(OUT_SHAPES / f"captions_{self.tick}.json", "w", encoding="utf-8") as f:
            json.dump(caps, f, ensure_ascii=False, indent=2)
        self.anneal_step += 1
        self.sigma = anneal_schedule(SIGMA0, GAMMA, self.anneal_step, SIGMA_MIN)
        return {"energetics": en, "caption": (caps["captions"][-1] if caps["captions"] else None)}

    async def run(self):
        while True:
            try:
                self.cube.step()
                self.tick += 1
                m = self.cube.metrics()
                self.mem.add_state(self.tick, m["tension"], m["energy"], m["size"])
                await self.bus.publish({"type": "metrics", "data": {"tick": self.tick, **m, "sigma": self.sigma}})
                if self.tick % AUTONOMOUS_INGEST_EVERY == 0:
                    await self.autonomous_ingest()
                if self.tick % REFLECT_EVERY == 0:
                    ref = make_reflection(self.tick, m)
                    self.mem.add_reflection(self.tick, ref)
                    await self.bus.publish({"type": "reflection", "data": {"tick": self.tick, "text": ref}})
                    r = ask_ollama_refine(m, ref)
                    adjust = r["adjust"]
                    self.mem.add_suggestion(self.tick, adjust)
                    self.cube.apply_adjustments(adjust)
                    await self.bus.publish({"type": "suggestion", "data": {"tick": self.tick, **adjust, "heuristic": not r.get("ok")}})
                    out = self._anneal_and_process()
                    if out:
                        await self.bus.publish({"type": "energetics", "data": {"tick": self.tick, **out["energetics"], "sigma": self.sigma}})
                        if out["caption"]:
                            await self.bus.publish({"type": "caption", "data": {"tick": self.tick, **out["caption"]}})
            except Exception as e:
                await self.bus.publish({"type": "error", "data": {"tick": self.tick, "error": str(e), "trace": traceback.format_exc()}})
            await asyncio.sleep(TICK_SEC)

