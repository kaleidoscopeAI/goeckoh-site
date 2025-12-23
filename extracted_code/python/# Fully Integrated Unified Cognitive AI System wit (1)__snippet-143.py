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
