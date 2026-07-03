FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY static/ ./static/

# Installer binaries are NOT bundled here — served via a private GitHub
# release (kaleidoscopeAI/goeckoh-releases), fetched on demand by /download.

# Persistent SQLite lives on a mounted volume at /data
RUN mkdir -p /data

ENV DATABASE_URL=sqlite:////data/goeckoh_licenses.db

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
