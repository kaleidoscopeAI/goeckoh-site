import asyncio
import aiohttp
from bs4 import BeautifulSoup
from transformers import pipeline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class VisualizationManager:
    def __init__(self, ai_system):
        self.ai_system = ai_system

    def generate_dashboard_json(self):
        n = len(self.ai_system.nodes)
        pos = np.array([node.position.components for node in self.ai_system.nodes])
        arousal = np.array([node.emotional_state.arousal for node in self.ai_system.nodes])

        fig = make_subplots(rows=2, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter'}], [{'type':'heatmap'}, None]])

        fig.add_trace(go.Scatter3d(x=pos[:,0], y=pos[:,1], z=pos[:,2], mode='markers',
                                   marker=dict(color=arousal, colorscale='Viridis', size=5),
                                   name='Nodes'), row=1, col=1)

        fig.add_trace(go.Scatter(x=np.arange(n), y=arousal, mode='lines', name='Arousal'), row=1, col=2)

        fig.add_trace(go.Heatmap(z=arousal.reshape(int(np.sqrt(n)), -1), colorscale='Viridis'), row=2, col=1)

        return fig.to_json()

class MemoryCrystal:
    def __init__(self):
        self.storage = []

    async def ingest_text(self, text, sentiment):
        # Placeholder for embedding & storage logic
        print(f"Ingested Text: {text[:80]}... Sentiment: {sentiment}")

class AsyncCrawler:
    def __init__(self, memory_crystal, concurrency=5):
        self.memory_crystal = memory_crystal
        self.semaphore = asyncio.Semaphore(concurrency)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.sentiment = pipeline("sentiment-analysis")

    async def fetch(self, session, url):
        async with self.semaphore:
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.text()
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")
        return None

    async def process_page(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = ' '.join(soup.stripped_strings)
        summary = self.summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        sent = self.sentiment(text[:512])[0]
        await self.memory_crystal.ingest_text(summary, sent)
        print(f"Processed page. Summary: {summary} Sentiment: {sent}")

    async def crawl(self, start_urls):
        async with aiohttp.ClientSession() as session:
            for url in start_urls:
                html = await self.fetch(session, url)
                if html:
                    await self.process_page(html)
