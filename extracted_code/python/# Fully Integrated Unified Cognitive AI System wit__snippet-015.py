import aiohttp
import asyncio
from bs4 import BeautifulSoup
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis")

class AdvancedCrawler:
    def __init__(self, memory_crystal, concurrency=5, max_urls=50):
        self.memory_crystal = memory_crystal
        self.semaphore = asyncio.Semaphore(concurrency)
        self.visited = set()
        self.max_urls = max_urls

    async def fetch(self, session, url):
        async with self.semaphore:
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        return await response.text()
            except Exception as e:
                print(f"Error fetching {url}: {e}")
        return None

    async def extract_text(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = ' '.join(soup.stripped_strings)
        return text

    async def process_page(self, url, session):
        html = await self.fetch(session, url)
        if html is None:
            return
        text = await self.extract_text(html)
        summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        sentiment = sentiment_analyzer(text[:512])[0]
        # Add to memory crystal embeddings or indexed storage
        await self.memory_crystal.ingest_text(summary, sentiment)
        print(f"Crawled URL: {url}\nSummary: {summary}\nSentiment: {sentiment}")

    async def crawl(self, start_urls):
        async with aiohttp.ClientSession() as session:
            queue = list(start_urls)
            while queue and len(self.visited) < self.max_urls:
                url = queue.pop(0)
                if url in self.visited:
                    continue
                self.visited.add(url)
                await self.process_page(url, session)
                # Optional: extract links for recursive crawl here
