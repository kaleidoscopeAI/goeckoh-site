import aiohttp
import asyncio
from bs4 import BeautifulSoup
from transformers import pipeline

# Setup HuggingFace summarizer and sentiment-analyzer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment = pipeline("sentiment-analysis")

class AdvancedWebCrawler:

    def __init__(self, max_concurrent=5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.visited = set()

    async def fetch_page(self, session, url):
        async with self.semaphore:
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.text()
            except Exception as e:
                print(f"Fetch failure {url}: {e}")
        return None

    async def extract_text(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = ' '.join(soup.stripped_strings)
        return text

    async def process_text(self, text):
        summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        sentiment_result = sentiment(text[:512])[0]  # Limit text size for sentiment model
        return summary, sentiment_result

    async def crawl(self, start_urls):
        async with aiohttp.ClientSession() as session:
            for url in start_urls:
                if url in self.visited:
                    continue
                self.visited.add(url)
                html = await self.fetch_page(session, url)
                if not html:
                    continue
                text = await self.extract_text(html)
                summary, sentiment_result = await self.process_text(text)
                print(f"Crawled: {url}\nSummary: {summary}\nSentiment: {sentiment_result}")

                # Feed summary or embeddings into memory crystal here
                # e.g., self.memory_crystal.add(summary, sentiment_result)

                # Optionally extract links for recursive crawl (respect robots.txt)
