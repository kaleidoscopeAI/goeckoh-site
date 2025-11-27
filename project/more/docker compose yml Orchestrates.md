# docker-compose.yml
# Orchestrates the UNI system core deployment.

version: '3.8'

services:
  uni-core:
    build: .
    # Ports required for external access (e.g., from a frontend)
    ports:
      - "8000:8000"
      # Ollama typically runs on 11434, expose this if external LLM interaction is needed
      # - "11434:11434"
    volumes:
      # Volume for persistent Ollama models and potential data/logs
      - uni_data:/app/data
    restart: unless-stopped
    # NOTE: The HIDController logic requires elevated permissions on the host system
    # to access /dev/hidg0. This may require run parameters like 'privileged: true'
    # or specific device mappings (devices: ['/dev/hidg0:/dev/hidg0']) in a real 
    # embedded/host environment. They are omitted here for general deployment safety.
    
volumes:
  uni_data:

