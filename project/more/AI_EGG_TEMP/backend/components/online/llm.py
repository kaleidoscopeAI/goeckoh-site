# backend/components/online/llm.py
import httpx
import logging
from interfaces import BaseLLM

class OnlineLLM(BaseLLM):
    def __init__(self, llm_config):
        self.model = llm_config.get('online_model', 'llama3:8b')
        self.server_url = llm_config.get('server_url', 'http://localhost:11434')

    async def reflect(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
                response = await client.post(f"{self.server_url}/api/generate", json=payload, timeout=60)
                return response.json()['response']
        except Exception as e:
            logging.error(f"Ollama reflection error: {e}")
            return "(Inner voice is silent due to a network error.)"
