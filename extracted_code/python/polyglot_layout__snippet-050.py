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

# API
app = FastAPI(title="Seed-Crystal AGI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
orch = Orchestrator()

@app.on_event("startup")
async def boot():
  asyncio.create_task(orch.run())

@app.get("/status")
def status():
  return {"ok": True, "state": orch.snapshot()}

@app.get("/recent")
def recent(table: str = Query("states"), limit: int = 50):
  return {"ok": True, "rows": orch.mem.recent(table, limit)}

@app.post("/ingest")
def ingest(url: str):
  title, text = fetch_url(url)
  doc_id = orch.mem.add_doc_with_embed(url, title, text)
  return {"ok": True, "doc_id": doc_id}

@app.get("/", response_class=HTMLResponse)
def home():
  # Inline UI similar to provided, but simplified for brevity.
  return "<html><body><h1>Seed-Crystal AGI</h1><p>Access /status or /ws for real-time.</p></body></html>"

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
  await ws.accept()
  q = orch.bus.subscribe()
  try:
     await ws.send_text(json.dumps({"type": "hello", "data": orch.snapshot()}))
     while True:
       msg = await q.get()
       await ws.send_text(json.dumps(msg))

