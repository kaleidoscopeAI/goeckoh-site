# backend/components/offline/crawler.py
import os
import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from interfaces import BaseCrawler
from components.offline.embedder import LocalEmbedder

@dataclass
class Document:
    url: str
    text: str
    embedding: Optional[np.ndarray] = None
    links: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    score: float = 0.0

def ensure_corpus(path='./corpus'):
    os.makedirs(path, exist_ok=True)
    files = [f for f in os.listdir(path) if f.endswith('.txt')]
    if not files:
        samples = {
            'ai.txt': 'Artificial intelligence studies algorithms that learn from data.',
            'cognition.txt': 'Cognition explores perception, attention, memory, and reasoning.',
            'agent.txt': 'An agent observes, decides and acts in an environment to achieve goals.'
        }
        for name, text in samples.items():
            with open(os.path.join(path, name), 'w') as fh:
                fh.write(text)
        logging.info('Sample corpus created in ./corpus')

class OfflineCrawler(BaseCrawler):
    def __init__(self, ai_core, corpus_dir='./corpus', embedder: Optional[LocalEmbedder]=None):
        self.ai = ai_core
        self.corpus_dir = corpus_dir
        self.queue = deque()
        self.visited = set()
        self.embedder = embedder or LocalEmbedder(n_components=64)
        self.running = False
        ensure_corpus(self.corpus_dir)
        self._load_corpus()

    def _load_corpus(self):
        files = [os.path.join(self.corpus_dir, f) for f in os.listdir(self.corpus_dir) if f.endswith('.txt')]
        texts = []
        for f in files:
            with open(f, 'r', encoding='utf-8') as fh:
                texts.append(fh.read())
        if texts:
            self.embedder.fit(texts)
        for f in files:
            self.queue.append('file://' + os.path.abspath(f))

    def _read_doc(self, url: str) -> Optional[Document]:
        if not url.startswith('file://'):
            return None
        path = url[len('file://'):]
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as fh:
            text = fh.read()
        return Document(url=url, text=text)

    async def crawl_once(self, max_docs=5):
        processed = 0
        while self.queue and processed < max_docs:
            url = self.queue.popleft()
            if url in self.visited:
                continue
            doc = self._read_doc(url)
            if not doc:
                continue
            doc.embedding = self.embedder.embed(doc.text)
            doc.score = float(np.random.rand())
            self.visited.add(url)
            await self.ai.on_new_document(doc.__dict__)
            processed += 1
        return processed

    async def start(self, interval=5.0):
        self.running = True
        logging.info('LocalCrawler started')
        while self.running:
            try:
                n = await self.crawl_once(max_docs=3)
                if n == 0:
                    await asyncio.sleep(interval)
                else:
                    await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f'Crawler error: {e}')
                await asyncio.sleep(2.0)

    def stop(self):
        self.running = False
        logging.info('LocalCrawler stopped')
