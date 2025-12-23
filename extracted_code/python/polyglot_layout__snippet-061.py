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

