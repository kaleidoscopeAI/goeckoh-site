import aiohttp
import asyncio
import hashlib
from bs4 import BeautifulSoup
from transformers import pipeline

class AdvancedAsyncCrawler:
    def __init__(self, memory_crystal, concurrency=10):
        self.memory_crystal = memory_crystal
        self.semaphore = asyncio.Semaphore(concurrency)
        self.visited = set()
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.sentiment = pipeline("sentiment-analysis")

    async def fetch(self, session, url):
        async with self.semaphore:
            if url in self.visited:
                return None
            self.visited.add(url)
            try:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        return await resp.text()
            except Exception:
                return None

    async def extract_and_process(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        for s in soup(["script", "style"]):
            s.decompose()
        text = soup.get_text(separator=' ', strip=True)
        summary = self.summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        sent = self.sentiment(text[:512])[0]
        await self.memory_crystal.add(summary)

    async def crawl(self, start_urls):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch(session, url) for url in start_urls]
            pages = await asyncio.gather(*tasks)
            process_tasks = [self.extract_and_process(html) for html in pages if html]
            await asyncio.gather(*process_tasks)
