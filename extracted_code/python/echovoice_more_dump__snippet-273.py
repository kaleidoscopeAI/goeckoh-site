import requests
import json

def reflect_with_ollama(self, prompt: str) -> str:
    """Sends a state summary to a live Ollama instance and returns its reflection."""
    print(f"\n[Cognitive Engine] Sending prompt to live Ollama instance...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate", # Default Ollama API endpoint
            json={
                "model": "llama3", # Or your preferred model
                "prompt": prompt,
                "stream": False
            },
            timeout=30 # Set a timeout for the request
        )
        response.raise_for_status() # Raise an exception for bad status codes
        
        # The actual text is in the 'response' key of the JSON object
        response_text = json.loads(response.text).get("response", "No response from model.")
        
        print(f"[Cognitive Engine] Ollama reflection received.")
        return response_text.strip()
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Could not connect to Ollama: {e}")
        return "Error: Could not generate reflection. Is the Ollama server running?"

