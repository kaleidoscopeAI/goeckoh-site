import wikipediaapi
import networkx as nx
import matplotlib.pyplot as plt

class WikiCrawler:
    def __init__(self, start_page):
        self.wiki = wikipediaapi.Wikipedia('en')
        self.start_page = start_page
        self.graph = nx.Graph()
    
    def crawl(self, depth=2):
        """Crawl Wikipedia starting from the given page."""
        self._crawl_page(self.start_page, depth)
    
    def _crawl_page(self, page_name, depth):
        if depth == 0:
            return
        page = self.wiki.page(page_name)
        if not page.exists():
            print(f"Page {page_name} does not exist.")
            return
        self.graph.add_node(page_name)
        print(f"Crawling: {page_name}")
        for link in list(page.links.keys())[:10]:  # Limit to 10 links per page
            self.graph.add_edge(page_name, link)
            self._crawl_page(link, depth - 1)
    
    def visualize(self):
        """Visualize the Wikipedia graph."""
        plt.figure(figsize=(10, 8))
        nx.draw(self.graph, with_labels=True, node_size=500, font_size=8, node_color="skyblue")
        plt.title("Wikipedia Crawler Network")
        plt.show()

if __name__ == "__main__":
    crawler = WikiCrawler("Artificial intelligence")
    crawler.crawl(depth=2)
    crawler.visualize()

