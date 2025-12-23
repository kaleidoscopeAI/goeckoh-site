import aiohttp
from .client import LLMClient
class OllamaClient(LLMClient):
def __init__(self, model="mistral", host="http://localhost:11434/api/generate", **kwargs):
