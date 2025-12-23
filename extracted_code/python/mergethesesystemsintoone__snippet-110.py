class DataProcessor:
    def __init__(self):
        self.pool = Pool(processes=os.cpu_count() // 2)  # Parallel for heavy ops
        self.word2vec = None  # Lazy init

    def process_text(self, text: str) -> Dict:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        topics = self._identify_topics([[t.text for t in doc]])
        return {"entities": entities, "topics": topics}

    def process_image(self, img_url: str) -> Dict:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert('L')
        array = np.array(img)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        edges = self._convolve(array, sobel_x)
        shapes = self._detect_shapes(edges)
        return {"shapes": shapes}

    def process_numerical(self, data: List[float]) -> Dict:
        fft = np.fft.fft(data)
        peaks = np.argwhere(np.abs(fft) > np.mean(np.abs(fft)) + np.std(np.abs(fft))).flatten().tolist()
        return {"peaks": peaks}

    def _convolve(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        from scipy.signal import convolve2d  # Use scipy for speed
        return convolve2d(img, kernel, mode='same')

    def _detect_shapes(self, edges: np.ndarray) -> List:
        visited = np.zeros_like(edges, dtype=bool)
        shapes = []
        for i in range(0, edges.shape[0], 2):  # Stride for speed
            for j in range(0, edges.shape[1], 2):
                if edges[i,j] > 0 and not visited[i,j]:
                    contour = self._dfs_contour(edges, visited, i, j)
                    if len(contour) > 5:  # Min size
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
            if len(contour) > MAX_HISTORY // 10: break  # Per contour limit
        return contour

    def _identify_topics(self, sentences: List[List[str]], num_topics: int = 3) -> List[List[str]]:
        from scipy.sparse import lil_matrix
        words = list(set(w for sent in sentences for w in sent))
        if not words: return []
        w2idx = {w: i for i, w in enumerate(words)}
        matrix = lil_matrix((len(words), len(words)))
        for sent in sentences:
            for i in range(len(sent)):
                for j in range(i+1, len(sent)):
                    w1, w2 = sorted([sent[i], sent[j]])
                    matrix[w2idx[w1], w2idx[w2]] += 1
        matrix = matrix.tocsr()
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=num_topics)
        U = svd.fit_transform(matrix)
        topics = [[] for _ in range(num_topics)]
        for i in range(len(words)):
            topic_idx = np.argmax(np.abs(U[i]))
            topics[topic_idx].append(words[i])
        return topics

