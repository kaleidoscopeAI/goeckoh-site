  def __init__(self, mem:'Memory'): self.mem=mem
  def topk(self, query:str, k:int=6)->Tuple[List[int], List[float]]:
      E, ids = self.mem.embeddings(max_items=512)
      if E.size==0: return [], []
      qv = embed_text(query)
      sims = (E @ qv) / (np.linalg.norm(E,axis=1) * (np.linalg.norm(qv)+1e-9) + 1e-12)
      order = np.argsort(-sims)[:k]
      return [ids[i] for i in order], [float(sims[i]) for i in order]

