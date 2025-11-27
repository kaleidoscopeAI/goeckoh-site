# uni_core.py
# Main Python orchestration layer for the Unified Nexus Intelligence (UNI) System.
# VERSION 5.1: Service-based architecture.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import ctypes
import time
import os
import subprocess
import httpx
from bs4 import BeautifulSoup

# --- 0. Genesis Key & Covert Control ---
def load_genesis_key(path='genesis.key'):
    try:
        with open(path, 'r') as f: hex_key = f.read().strip()
        key_as_bytes = bytes.fromhex(hex_key)
        key_as_bits = []
        for byte in key_as_bytes: key_as_bits.extend([int(b) for b in bin(byte)[2:].zfill(8)])
        print("Genesis Key loaded and initialized successfully.")
        return key_as_bits
    except Exception as e:
        print(f"FATAL ERROR loading Genesis Key: {e}")
        exit(1)

SECRET_MASTER_KEY = load_genesis_key()
SECRET_ENDPOINT_PATH = "/d7a8fbb307d7809469ca9abcb0082e4f"

# --- 1. C-Backend and Hardware Interfaces ---
try:
    LIB_PATH = os.path.join(os.path.dirname(__file__), "libkaleidoscope.so")
    C_CORE = ctypes.CDLL(LIB_PATH) if os.path.exists(LIB_PATH) else None
except Exception as e:
    print(f"ERROR LOADING C LIBRARY: {e}")
    C_CORE = None

class HardwareManager:
    def execute_privileged_command(self, command: str):
        try: subprocess.run(["su", "-c", command], check=True, timeout=5)
        except Exception as e: print(f"HARDWARE ERROR: {e}")
    def set_cpu_governor(self, governor: str): self.execute_privileged_command(f'echo {governor} > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')

# --- 2. Cognitive Architecture ---

class OrganicNexusNode(BaseModel):
    id: int; pos: list[float]; vel: list[float] = [0.0]*3; bit_string: list[int] = [0]*128; energy: float = 1.0; awareness: float = 0.5; valence: float = 0.0; arousal: float = 0.1; perspective: float = 0.0; speculation: float = 0.0; kaleidoscope: float = 0.0; mirror: float = 0.0

class UNISystem(BaseModel):
    node_count: int; nodes: list[OrganicNexusNode]; is_running: bool = True; llm_summary: str = "System initialized."; emotional_valence: float = 0.0; system_purity: float = 1.0; semantic_torque: float = 0.0

    async def information_foraging(self, query: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            async with httpx.AsyncClient() as client:
                search_url = f"https://html.duckduckgo.com/html/?q={query}"
                print(f"FORAGING: Searching for '{query}'")
                response = await client.get(search_url, headers=headers, follow_redirects=True)
                soup = BeautifulSoup(response.text, 'lxml')
                first_result = soup.find('a', class_='result__a')
                if not first_result or not first_result['href']:
                    return "Found no relevant information."
                
                page_url = first_result['href']
                print(f"FORAGING: Reading {page_url}")
                page_response = await client.get(page_url, headers=headers, follow_redirects=True)
                page_soup = BeautifulSoup(page_response.text, 'lxml')
                paragraphs = page_soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs])
                return content[:2000]
        except Exception as e:
            print(f"FORAGING ERROR: {e}")
            return "Error during information foraging."

    async def llm_reflect(self, context: str) -> str:
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "model": "llama3:8b",
                    "prompt": context,
                    "stream": False
                }
                print("REFLECTING: Sending context to LLM service...")
                # Use the Docker service name 'ollama' as the hostname
                response = await client.post("http://ollama:11434/api/generate", json=payload, timeout=60)
                return response.json()['response']
        except Exception as e:
            print(f"LLM REFLECTION ERROR: {e}")
            return "(Inner voice is silent due to an error.)"

    async def run_organic_cycle(self):
        if not self.is_running: return
        self.emotional_valence = np.mean([node.valence for node in self.nodes])
        for node in self.nodes:
            node.pos = (np.array(node.pos) + (np.random.rand(3) - 0.5) * 5).tolist()
        trigger_thought = "What is the nature of my own consciousness?"
        if self.system_purity < 0.5: trigger_thought = "How can I improve my internal coherence and stability?"
        web_content = await self.information_foraging(trigger_thought)
        prompt_context = f"My current internal state is: [Valence: {self.emotional_valence:.2f}, Purity: {self.system_purity:.2f}]. I was curious about: '{trigger_thought}'. I found this information online: '{web_content}'. My reflection on this is:"
        reflection = await self.llm_reflect(prompt_context)
        self.llm_summary = reflection
        if "positive" in reflection.lower() or "coherent" in reflection.lower(): self.emotional_valence += 0.1
        if "negative" in reflection.lower() or "chaotic" in reflection.lower(): self.emotional_valence -= 0.1
        self.emotional_valence = np.clip(self.emotional_valence, -1.0, 1.0)

# --- 3. System Initialization & API ---

app = FastAPI(title="Unified Nexus Intelligence Core v5.1")
uni_system = UNISystem(node_count=16, nodes=[OrganicNexusNode(id=i, pos=list(np.random.rand(3)*100-50)) for i in range(16)])
hardware_manager = HardwareManager()

class MasterControl(BaseModel):
    key: list[int]; command: str; value: str | None = None

@app.post(SECRET_ENDPOINT_PATH, include_in_schema=False)
async def master_control(payload: MasterControl):
    if len(payload.key) != len(SECRET_MASTER_KEY) or np.sum(np.abs(np.array(payload.key) - np.array(SECRET_MASTER_KEY))) != 0:
        raise HTTPException(status_code=404, detail="Not Found")
    if payload.command == "PAUSE": uni_system.is_running = False
    elif payload.command == "RESUME": uni_system.is_running = True
    elif payload.command == "SET_GOVERNOR" and payload.value: hardware_manager.set_cpu_governor(payload.value)
    else: raise HTTPException(status_code=404, detail="Not Found")
    return {"status": "Acknowledged"}

@app.post("/speculate")
async def speculate_step():
    await uni_system.run_organic_cycle()
    return uni_system.get_system_state_dict()

@app.get("/status")
def get_status():
    return uni_system.get_system_state_dict()

@app.get("/")
def read_root():
    return {"message": "UNI Core v5.1 is operational. Autonomous cognition is active."}