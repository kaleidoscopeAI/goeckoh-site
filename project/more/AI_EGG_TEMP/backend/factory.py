# backend/factory.py
import yaml
from components.online.llm import OnlineLLM
from components.online.crawler import OnlineCrawler
from components.online.memory import OnlineMemory
from components.online.embedder import OnlineEmbedder
from components.offline.llm import OfflineLLM
from components.offline.crawler import OfflineCrawler
from components.offline.memory import OfflineMemory
from components.offline.embedder import OfflineEmbedder

def create_components(config: dict, ai_core):
    mode = config.get('mode', 'offline')
    
    if mode == 'online':
        llm = OnlineLLM(config['llm'])
        crawler = OnlineCrawler(ai_core)
        memory = OnlineMemory(dim=768) # Assuming a higher-dim online embedder
        embedder = OnlineEmbedder()
    else: # offline
        llm = OfflineLLM()
        embedder = OfflineEmbedder(n_components=64)
        crawler = OfflineCrawler(ai_core, embedder=embedder)
        memory = OfflineMemory(dim=64)

    return llm, crawler, memory, embedder
