import random
from textwrap import dedent
class MockLLMClient:
async def query(self, prompt: str) -> str:
