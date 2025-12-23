  def __init__(self, mem:Memory): self.mem=mem
  def topk(self, query:str, k:int=6)->Tuple[List[int], List[float]]:
      E, ids = self.mem.embeddings(max_items=512)
      if E.size==0: return [], []

