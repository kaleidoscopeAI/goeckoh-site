# Dockerfile
# Builds the Unified Nexus Intelligence (UNI) core environment.

FROM python:3.12-slim

# Install compilation tools for the C core and other necessary system utilities
RUN apt-get update && \
    apt-get install -y gcc build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the core source files
COPY requirements.txt .
COPY uni_core.py .
COPY kaleidoscope_core.c .

# 1. Compile the C core into a shared library
# This command compiles the C code into libkaleidoscope.so, which Python will load.
RUN gcc -shared -o libkaleidoscope.so -fPIC kaleidoscope_core.c

# 2. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 3. Setup Ollama and pull Llama 3 model (as per the architectural documents)
# Ollama runs as a background process within the container
ENV OLLAMA_HOST=0.0.0.0
RUN curl -L https://ollama.com/download/install.sh | sh
RUN /usr/local/bin/ollama serve & \
    /usr/local/bin/ollama pull llama3:8b && \
    fg

# 4. Set the command to run the FastAPI server (Uvicorn)
# The server will run on port 8000
CMD ["uvicorn", "uni_core:app", "--host", "0.0.0.0", "--port", "8000"]

