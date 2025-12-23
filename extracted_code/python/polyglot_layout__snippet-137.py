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
               await self.bus.publish({"type":"metrics","data":{"tick":self.tick, **m, "sigma": self.sigma, "H_bits":self.hbits,
