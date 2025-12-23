# Dockerfile
# Builds the Unified Nexus Intelligence (UNI) core environment (v2).

FROM python:3.12-slim

# Install compilation tools for C core, web scraping, and other utilities
RUN apt-get update && \
    apt-get install -y gcc build-essential git curl libxml2-dev libxslt-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the core source files
COPY . . 

# 1. Compile the C core into a shared library
RUN gcc -shared -o libkaleidoscope.so -fPIC kaleidoscope_core.c

# 2. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 3. Set the command to run the FastAPI server
# Ollama is now a separate service and does not need to be started here.
CMD ["uvicorn", "uni_core:app", "--host", "0.0.0.0", "--port", "8000"]

