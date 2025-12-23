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

# Memory Store (with Redis fallback to dict if fail)

class MemoryStore:

  def __init__(self, path: str, redis: Optional[aioredis.Redis]):

    self.con = sqlite3.connect(path, check_same_thread=False)

    self.cur = self.con.cursor()

    self.cur.execute("CREATE TABLE IF NOT EXISTS dna (gen INTEGER PRIMARY KEY, dna TEXT)")

    self.cur.execute("CREATE INDEX IF NOT EXISTS idx_gen ON dna(gen)")

    self.cur.execute("CREATE TABLE IF NOT EXISTS insights (id TEXT PRIMARY KEY, data TEXT)")

    self.cur.execute("CREATE INDEX IF NOT EXISTS idx_id ON insights(id)")

    self.cur.execute("CREATE TABLE IF NOT EXISTS graph (source TEXT, target TEXT, weight REAL)")

    self.cur.execute("CREATE INDEX IF NOT EXISTS idx_source ON graph(source)")

    self.con.commit()

    self.redis = redis

    self.fallback_cache = {} # Dict fallback

  async def _get_cache(self, key: str):

