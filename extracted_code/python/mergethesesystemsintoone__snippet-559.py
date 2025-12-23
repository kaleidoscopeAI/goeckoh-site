def __init__(self):
    self.pool = Pool(processes=os.cpu_count() // 2)

async def process_text_batch(self, texts: List[str]) -> List[Dict]:
    return await asyncio.get_event_loop().run_in_executor(None, self._process_text_batch_sync, texts)

def _process_text_batch_sync(self, texts: List[str]) -> List[Dict]:
    docs = list(nlp.pipe(texts, batch_size=BATCH_SIZE))
    results = self.pool.map(self._text_worker, docs)
    return results

def _text_worker(self, doc) -> Dict:
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    sentences = [[t.text for t in sent] for sent in doc.sents]
    topics = self._identify_topics(sentences)
    return {"entities": entities, "topics": topics}

def process_image_batch(self, img_urls: List[str]) -> List[Dict]:
    return self.pool.map(self._process_image_sync, img_urls)

def _process_image_sync(self, img_url: str) -> Dict:
    try:
        response = requests.get(img_url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert('L')
        array = np.array(img)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
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
        if len(contour) > MAX_HISTORY // 10: break
    return contour

def _identify_topics(self, sentences: List[List[str]], num_topics: int = 3) -> List[List[str]]:
    from scipy.sparse import lil_matrix
    words = list(set(w for sent in sentences for w in sent))
    if not words: return []
    w2idx = {w: i for i, w in enumerate(words)}
    matrix = lil_matrix((len(words), len(words)))
    for sent in sentences:
        sent_idx = [w2idx[w] for w in sent if w in w2idx]
        for i in range(len(sent_idx)):
            for j in range(i+1, len(sent_idx)):
                w1, w2 = sorted([sent_idx[i], sent_idx[j]])
                matrix[w1, w2] += 1
    matrix = matrix.tocsr()
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=num_topics)
    U = svd.fit_transform(matrix)
    topics = [[] for _ in range(num_topics)]
    for i in range(len(words)):
        topic_idx = np.argmax(np.abs(U[i]))
        topics[topic_idx].append(words[i])
    return topics

