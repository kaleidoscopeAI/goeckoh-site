                  # drive avatar
                  self.avatar.step(en, dt=0.05)
                  img_path = self.avatar.render(self.state.tick)
                  await self.bus.pub({"type":"avatar","data":{"tick":self.state.tick, "frame":str(img_path)}})
          except Exception as e:
              await self.bus.pub({"type":"error","data":{"tick":self.state.tick,"error":str(e),"trace":traceback.format_exc()}})
          await asyncio.sleep(TICK_SEC)

  # interactive thinking stays available but not required
  async def think(self, text:str)->Dict[str,Any]:
      try:
          langs = [str(l) for l in detect_langs(text)]
      except Exception:
          langs = [detect(text)] if text.strip() else ["en"]
      lang = (langs[0].split(":")[0] if langs else "en").lower()
      retr = Retriever(self.mem)
      top_ids, top_sims = retr.topk(text, k=8)
      async def math_task():
          ok,res = MathSolver.solve_expr(text)
          return {"ok":ok, "res":res, "weight": 0.9 if ok else 0.0, "tag":"math"}
      async def logic_task():
          plan = LogicPlanner.plan(text)
          return {"ok":True, "res":"; ".join(plan), "weight":0.6, "tag":"plan"}
      async def compose_task():
          pieces=["Synthesis:"]
          if any(k in text.lower() for k in ["why","because","explain","how"]):
              pieces.append("Explaining step-by-step, then summarizing.")
          else:
              pieces.append("Combining relevant facts with a logical sequence.")
          return {"ok":True,"res":" ".join(pieces),"weight":0.5,"tag":"compose"}
      r_math, r_plan, r_comp = await asyncio.gather(math_task(), logic_task(), compose_task())
      best = max([r for r in [r_math,r_plan,r_comp] if r["ok"]], key=lambda r:r["weight"], default={"res":"(no
