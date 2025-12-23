import requests
from bs4 import BeautifulSoup

class WikiCrawler:
    def __init__(self, topic):
        self.topic = topic
        self.base_url = "https://en.wikipedia.org/wiki/"
        self.data = {}

    def fetch_data(self):
        """Fetch and parse data from Wikipedia."""
        try:
            url = self.base_url + self.topic.replace(" ", "_")
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")
            self.data = {f"Paragraph-{i}": p.get_text() for i, p in enumerate(paragraphs)}
        except Exception as e:
            print(f"Error fetching data: {e}")
        return self.data

