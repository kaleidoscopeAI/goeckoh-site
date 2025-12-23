      for u,v,data in self.G.edges(data=True):
          b: Bond = data['bond']
          pu = self.G.nodes[u]['node'].pos; pv = self.G.nodes[v]['node'].pos
          d = pv - pu; L = float(np.linalg.norm(d) + 1e-8)
          F = b.k * (L - b.rest) * (d / L)
          forces[u] += F; forces[v] -= F
      for i, data in self.G.nodes(data=True):
          n: Node = data['node']
          if n.fixed: continue
          n.pos += dt * forces[i]; n.pos *= damp
      self.tick += 1

  def metrics(self) -> Dict[str, float]:
      tension=0.0; energy=0.0
      for u,v,data in self.G.edges(data=True):
          b: Bond = data['bond']
          pu = self.G.nodes[u]['node'].pos; pv = self.G.nodes[v]['node'].pos
          L = float(np.linalg.norm(pv - pu))
          tension += abs(L - b.rest)
          energy += 0.5 * b.k * (L - b.rest)**2
      m = max(1, self.G.number_of_edges())
      return {"tension": tension/m, "energy": energy/m, "size": self.G.number_of_nodes()}

