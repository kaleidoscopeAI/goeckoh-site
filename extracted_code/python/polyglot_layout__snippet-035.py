       edges = self._convolve(array, sobel_x)

       shapes = self._detect_shapes(edges)

       return {"shapes": shapes}

    except Exception:

       return {"error": "Image fetch failed"}

  def process_numerical_batch(self, data_lists: List[List[float]]) -> List[Dict]:

    return self.pool.map(self._process_numerical_sync, data_lists)

  def _process_numerical_sync(self, data: List[float]) -> Dict:

    fft = np.fft.fft(data)

    peaks = np.argwhere(np.abs(fft) > np.mean(np.abs(fft)) + np.std(np.abs(fft))).flatten().tolist()

    return {"peaks": peaks}

  def _convolve(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    from scipy.signal import convolve2d

    return convolve2d(img, kernel, mode='same')

  def _detect_shapes(self, edges: np.ndarray) -> List:

    visited = np.zeros_like(edges, dtype=bool)

    shapes = []

    strides = 2

    for i in range(0, edges.shape[0], strides):

       for j in range(0, edges.shape[1], strides):

         if edges[i,j] > 0 and not visited[i,j]:

            contour = self._dfs_contour(edges, visited, i, j)

            if len(contour) > 10:

               shapes.append({"vertices": len(contour)})

    return shapes

  def _dfs_contour(self, edges: np.ndarray, visited: np.ndarray, x: int, y: int) -> List[Tuple[int,int]]:

    stack = deque([(x, y)])

    contour = []

    while stack:

       cx, cy = stack.pop()

       if visited[cx, cy]: continue

       visited[cx, cy] = True

       contour.append((cx, cy))

       for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:

         nx, ny = cx + dx, cy + dy

         if 0 <= nx < edges.shape[0] and 0 <= ny < edges.shape[1] and edges[nx,ny] > 0 and not visited[nx,ny]:

            stack.append((nx, ny))


