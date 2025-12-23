  def __init__(self, n_per_edge: int = 6, seed: int = 42):
      np.random.seed(seed); self.G = nx.Graph(); self.tick = 0; idc = 0
      for x in range(n_per_edge):
          for y in range(n_per_edge):
              for z in range(n_per_edge):
                  p = np.array([x, y, z], dtype=float); p = 2 * (p / (n_per_edge - 1)) - 1
                  self.G.add_node(idc, node=Node(id=idc, pos=p, fixed=False)); idc += 1
      def idx(x, y, z): return x * (n_per_edge**2) + y * n_per_edge + z
      for x in range(n_per_edge):
          for y in range(n_per_edge):
              for z in range(n_per_edge):
                  u = idx(x, y, z)
                  for dx, dy, dz in [(1,0,0),(0,1,0),(0,0,1)]:
                      nx_, ny_, nz_ = x+dx, y+dy, z+dz
                      if nx_ < n_per_edge and ny_ < n_per_edge and nz_ < n_per_edge:
                          v = idx(nx_, ny_, nz_)
                          self.G.add_edge(u, v, bond=Bond(a=u, b=v, k=0.15, rest=2 / (n_per_edge - 1)))
      corners = [0, idx(n_per_edge-1,0,0), idx(0,n_per_edge-1,0), idx(0,0,n_per_edge-1),
                 idx(n_per_edge-1,n_per_edge-1,0), idx(n_per_edge-1,0,n_per_edge-1),

