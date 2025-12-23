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
        caps = captions_from_shapes(DB_PATH, shapes, top_k=3, window=6, stride=6, hbits=self.hbits, sfield=self.sfield)
        if caps["captions"]:
            last = caps["captions"][-1]
            self.last_caption_text = last.get("caption", "") or self.last_caption_text
            self.mem.add_caption(self.tick, self.last_caption_text, last.get("top_ids", []), last.get("weights", []))

        with open(OUT_SHAPES / f"shapes_{self.tick}.json", "w", encoding="utf-8") as f:
            json.dump(shapes, f, ensure_ascii=False)
        with open(OUT_SHAPES / f"captions_{self.tick}.json", "w", encoding="utf-8") as f:
            json.dump(caps, f, ensure_ascii=False)

        self.anneal_step += 1
        self.sigma = anneal_schedule(SIGMA0, GAMMA, self.anneal_step, SIGMA_MIN)
        return {"energetics": en, "caption": (caps["captions"][-1] if caps["captions"] else None)}

    async def run(self):
        while True:
            try:
                self.cube.step(); self.tick += 1
                m = self.cube.metrics()
                self.mem.add_state(self.tick, m["tension"], m["energy"], m["size"])
                await self.bus.publish({"type":"metrics","data":{"tick":self.tick, **m, "sigma": self.sigma, "H_bits":self.hbits, "S_field":self.sfield}})

                # Autonomous ingest
                if self.tick % AUTONOMOUS_INGEST_EVERY == 0:
                    doc_id = self._ingest_local() or (self._ingest_online() if ALLOW_ONLINE else None)
                    if doc_id:
                        await self.bus.publish({"type":"ingest","data":{"doc_id":int(doc_id)}})

                # Reflection + adjust + anneal pipeline
                if self.tick % SC_REFLECT_EVERY == 0:
                    ref = make_reflection(self.tick, m); self.mem.add_reflection(self.tick, ref)
                    await self.bus.publish({"type":"reflection","data":{"tick":self.tick,"text":ref}})
                    r = ask_ollama_refine(m, ref)
                    adjust = r["adjust"]; self.mem.add_suggestion(self.tick, adjust)
                    # apply to cube
                    ks = float(adjust.get("k_scale",1.0)); rs = float(adjust.get("rest_scale",1.0))
                    ks = max(0.25, min(ks, 4.0)); rs = max(0.5, min(rs, 1.5))
                    for _,_,data in self.cube.G.edges(data=True):
                        b: Bond = data["bond"]; b.k *= ks; b.rest *= rs
                    await self.bus.publish({"type":"suggestion","data":{"tick":self.tick, **adjust, "heuristic": not r.get("ok")}})

                    out = self._anneal_and_process()
                    if out:
                        await self.bus.publish({"type":"energetics","data":{"tick":self.tick, **out["energetics"], "sigma": self.sigma}})
                        if out["caption"]:
                            await self.bus.publish({"type":"caption","data":{"tick":self.tick, **out["caption"]}})

                # Update avatar every tick regardless of UI
                self.avatar.update(self.hbits, self.sfield, self.last_caption_text)

            except Exception as e:
                await self.bus.publish({"type":"error","data":{"tick":self.tick, "error": str(e), "trace": traceback.format_exc()}})
            await asyncio.sleep(SC_TICK_SEC)

