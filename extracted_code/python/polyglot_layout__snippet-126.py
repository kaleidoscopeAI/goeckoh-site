  async def run(self):
      while True:
          try:
               self.cube.step(); self.tick += 1
               m = self.cube.metrics()
               self.mem.add_state(self.tick, m["tension"], m["energy"], m["size"])
               await self.bus.publish({"type":"metrics","data":{"tick":self.tick, **m, "sigma": self.sigma, "H_bits":self.hbits,
