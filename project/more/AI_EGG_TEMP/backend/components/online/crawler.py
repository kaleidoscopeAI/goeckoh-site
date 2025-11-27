# backend/components/online/crawler.py
import httpx
from bs4 import BeautifulSoup
import logging
from collections import deque
import asyncio
from interfaces import BaseCrawler

class OnlineCrawler(BaseCrawler):
    def __init__(self, ai_core):
        self.ai = ai_core
        self.queue = deque()
        self.visited = set()
        self.running = False

    async def start(self, interval: float):
        self.running = True
        logging.info('OnlineCrawler started')
        # Seed with a starting URL if empty
        if not self.queue:
            self.queue.append("https://en.wikipedia.org/wiki/Artificial_intelligence")

        while self.running:
            if not self.queue:
                await asyncio.sleep(interval)
                continue

            url = self.queue.popleft()
            if url in self.visited:
                continue

            try:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    logging.info(f"Crawling: {url}")
                    response = await client.get(url, timeout=20.0)
                    self.visited.add(url)

                    if "text/html" in response.headers.get("content-type", ""):
                        soup = BeautifulSoup(response.text, 'lxml')
                        text_content = ' '.join(p.get_text() for p in soup.find_all('p'))
                        doc = {'url': url, 'text': text_content}
                        await self.ai.on_new_document(doc)

                        # Add new links to queue
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            if href.startswith('http') and href not in self.visited:
                                self.queue.append(href)
            except Exception as e:
                logging.error(f"Error crawling {url}: {e}")
            
            await asyncio.sleep(interval) # Rate limit crawling

    def stop(self):
        self.running = False
        logging.info('OnlineCrawler stopped')
