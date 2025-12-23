from collections import deque
class WebCrawler:
    def __init__(self, allowed_domains):
        self.allowed=set(allowed_domains); self.queue=deque(); self.knowledge=[]
    def seed(self, urls):
        for u in urls: self.queue.append(u)
    def step(self, limit=3):
        fetched=[]
        while self.queue and len(fetched)<limit:
            u=self.queue.popleft()
            if any(d in u for d in self.allowed):
                doc={'url':u,'title':'Sample Doc','text':'Knowledge token '+u}
                self.knowledge.append(doc); fetched.append(doc)
        return fetched
