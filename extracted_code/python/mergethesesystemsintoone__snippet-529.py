def __init__(self):
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
    peaks = np.argwhere(np.abs(fft) > np.mean(np.abs(fft)) + np.std(np.abs(fft)))
    return {"peaks": peaks.tolist()}

def _convolve(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    output = np.zeros_like(img, dtype=float)
    pad = kernel.shape[0] // 2
    padded = np.pad(img, pad)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i, j] = np.sum(padded[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return output

def _detect_shapes(self, edges: np.ndarray) -> List:
    # Real contour finding (simple DFS)
    visited = np.zeros_like(edges, dtype=bool)
    shapes = []
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i,j] > 0 and not visited[i,j]:
                contour = self._dfs_contour(edges, visited, i, j)
                if len(contour) > 5:  # Min size
                    shapes.append({"vertices": len(contour)})
    return shapes

def _dfs_contour(self, edges: np.ndarray, visited: np.ndarray, x: int, y: int) -> List[Tuple[int,int]]:
    stack = [(x, y)]
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
    return contour

def _identify_topics(self, sentences: List[List[str]], num_topics: int = 3) -> List[List[str]]:
    co_occ = defaultdict(lambda: defaultdict(int))
    for sent in sentences:
        for i in range(len(sent)):
            for j in range(i+1, len(sent)):
                w1, w2 = sorted([sent[i], sent[j]])
                co_occ[w1][w2] += 1
    words = list(set(w for sent in sentences for w in sent))
    matrix = np.zeros((len(words), len(words)))
    w2idx = {w: i for i, w in enumerate(words)}
    for w1 in co_occ:
        for w2 in co_occ[w1]:
            matrix[w2idx[w1], w2idx[w2]] = co_occ[w1][w2]
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    topics = [[] for _ in range(num_topics)]
    for i in range(len(words)):
        topic_idx = np.argmax(np.abs(U[i, :num_topics]))
        topics[topic_idx].append(words[i])
    return topics

