def __init__(self):
    self.mem=Memory(DB_PATH); self.bus=Broadcaster()
    self.state=BrainState()
    self.rng=np.random.RandomState(101)

def _anneal_round(self):
    E, ids = self.mem.embeddings(max_items=192)
    if E.size==0: return None
    N=E.shape[0]
    edges = ring_edges(N, k=max(4,min(12,N-1)))
    S=np.zeros(N)
    for i in range(N):
        var=mc_var(E,i,k=min(8,N-1), sigma=self.state.sigma, M=4, rng=self.rng)
        S[i]=stability(var)
    en=energetics(E,S,edges,self.state.sigma)
    self.mem.log_energy(self.state.tick, self.state.sigma, en)
    maps=default_maps(en["H_bits"], en["S_field"], latency=0.2, fitness=max(0.0, 1.0-en["H_bits"]))
    sig=synth_signal(1.6, 22050, maps["a"], maps["m"], maps["rho"], maps["fc"])
    wav_path=OUT_AUDIO/f"onbrain_{self.state.tick}.wav"; write_wav_mono16(wav_path,22050,sig)
    X=stft_mag(np.array(sig,dtype=np.float64), sr=22050, win=1024, hop=256)
    V=head_features(X, make_bands(X.shape[0], H=4))
    # project "attention" to memory (no LLM)
    H,T,_=V.shape; D=E.shape[1]; d=24; rng=np.random.RandomState(1234)
    Wk=rng.normal(0, 1.0/math.sqrt(D), size=(D,d)); K=E@Wk; K/= (np.linalg.norm(K,axis=1,keepdims=True)+1e-9)
    captions=[]
    for h in range(H):
        Wq=rng.normal(0,1.0,size=(V.shape[2], d))
        Q=V[h]@Wq; Q/= (np.linalg.norm(Q,axis=1,keepdims=True)+1e-9)
        Satt=(Q@K.T)/(d*max(self.state.sigma, SIGMA_MIN))
        Satt -= Satt.max(axis=1, keepdims=True)
        P=np.exp(Satt); P/= (P.sum(axis=1,keepdims=True)+1e-12)
        svec=P.mean(axis=0); top=list(np.argsort(-svec)[:5])
        facts=self.mem.fact_text([ids[i] for i in top])
        cap="; ".join(facts.get(ids[i],"")[:80] for i in top if ids[i] in facts)
        if cap: captions.append(cap)
    if captions:
        self.mem.log_caption(self.state.tick, captions[-1], {"H_bits":en["H_bits"], "S_field":en["S_field"]})
    self.state.anneal_step += 1
    self.state.sigma = anneal_sigma(SIGMA0, GAMMA, self.state.anneal_step, SIGMA_MIN)
    return en, (captions[-1] if captions else "")

async def loop(self):
    while True:
        try:
            self.state.tick += 1
            if self.state.tick % REFLECT_EVERY == 0:
                out=self._anneal_round()
                if out:
                    en, cap = out
                    await self.bus.pub({"type":"energetics","data":{"tick":self.state.tick, **en, "sigma": self.state.sigma}})
                    if cap: await self.bus.pub({"type":"caption","data":{"tick":self.state.tick, "text":cap}})
        except Exception as e:
            await self.bus.pub({"type":"error","data":{"tick":self.state.tick,"error":str(e),"trace":traceback.format_exc()}})
        await asyncio.sleep(TICK_SEC)

# ---- Main thinking entry ----
async def think(self, text:str)->Dict[str,Any]:
    # 1) Detect languages (may be multiple)
    try:
        langs = [str(l) for l in detect_langs(text)]
    except Exception:
        langs = [detect(text)] if text.strip() else ["en"]
    lang = (langs[0].split(":")[0] if langs else "en").lower()

    # 2) Parallel domain solvers (no APIs)
    retr = Retriever(self.mem)
    top_ids, top_sims = retr.topk(text, k=8)
    facts = self.mem.fact_text(top_ids)
    ctx = [facts.get(i,"") for i in top_ids]

    async def math_task():
        ok,res = MathSolver.solve_expr(text)
        return {"ok":ok, "res":res, "weight": 0.9 if ok else 0.0, "tag":"math"}

    async def logic_task():
        plan = LogicPlanner.plan(text)
        # tiny critique: prefer steps that use retrieved context if any
        if ctx: plan = plan[:1]+["Review retrieved facts for relevance"]+plan[1:]
        return {"ok":True, "res":"; ".join(plan), "weight":0.6, "tag":"plan"}

    async def compose_task():
        # Compose an answer without LLMs: rule-based template over context + simple reasoning
        pieces=[]
        if ctx:
            pieces.append("Context:")
            for i,(cid,sim) in enumerate(zip(top_ids, top_sims)):
                t=facts.get(cid,"")
                if t: pieces.append(f"- [{i+1}] {t[:160]} (sim={sim:.3f})")
        pieces.append("Synthesis:")
        # very small heuristics
        if any(k in text.lower() for k in ["why","because","explain","how"]):
            pieces.append("I’ll explain step-by-step, then summarize.")
        elif any(k in text.lower() for k in ["solve","=", "integrate","differentiate","derivative","limit"]):
            pieces.append("See the math result and explanation above.")
        else:
            pieces.append("Combining the most relevant known facts with a logical sequence to address your request.")
        return {"ok":True,"res":"\n".join(pieces),"weight":0.5,"tag":"compose"}

    r_math, r_plan, r_comp = await asyncio.gather(math_task(), logic_task(), compose_task())

    # 3) Score & merge
    candidates=[r for r in [r_math,r_plan,r_comp] if r["ok"]]
    best = max(candidates, key=lambda r:r["weight"]) if candidates else {"res":"(no solver matched)", "tag":"none", "weight":0.0}
    label = label_for(lang)
    answer = f"{label}: {best['res']}"

    # 4) Log trace
    self.mem.log(self.state.tick, "think", {"lang":lang,"query":text,"selected":best["tag"]})
    await self.bus.pub({"type":"think","data":{"tick":self.state.tick,"lang":lang,"selected":best["tag"]}})

    return {
        "ok": True,
        "lang": lang,
        "selected": best["tag"],
        "answer": answer,
        "context_ids": top_ids,
        "context_sims": top_sims,
    }

