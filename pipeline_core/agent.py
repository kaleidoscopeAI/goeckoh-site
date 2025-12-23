import yaml
import jsonschema
import logging
from pathlib import Path
from document_loader import read_all_documents

SCHEMA_PATH = Path(__file__).parent / "config.schema.yaml"

class ValidationError(Exception):
    pass

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def validate_config(config: dict):
    schema = load_yaml(SCHEMA_PATH)
    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValidationError(str(e))
    return True

class Agent:
    def __init__(self, config):
        validate_config(config)
        self.system_prompt = config["system_prompt"]
        self.tools = config["tools"]
        self.tracing = config.get("tracing", False)
        log_level = config.get("log_level", "INFO")
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger("agent")
        if self.tracing:
            self.logger.info("Tracing enabled")
        # store ingested documents
        self.documents = []

    def respond(self, message: str) -> str:
        message = (message or "").strip()
        self.logger.debug("Received message: %s", message)
        if not message:
            return "No message provided."
        lower = message.lower()
        if "fix" in lower and "system" in lower:
            return "Acknowledged. I will attempt to repair common configuration issues."
        if "tools" in lower:
            tool_names = ", ".join([t["name"] for t in self.tools])
            return f"Available tools: {tool_names}"
        # Default deterministic echo response
        return f"Echo: {message}"

    def ingest_documents(self, documents_path: str, recursive: bool = True, extensions: list = None):
        """
        Ingest files matching 'extensions' from the provided folder.
        If recursive is True, also read subfolders.
        Returns a list of summaries: {path, length, snippet}
        """
        if not documents_path:
            raise RuntimeError("No documents_path provided for ingestion.")
        docs = read_all_documents(documents_path, extensions=extensions, recursive=recursive)
        self.documents = docs
        summaries = []
        for d in docs:
            text = d["text"]
            snippet = (text.strip()[:200] + ("..." if len(text) > 200 else "")) if text.strip() else ""
            summaries.append({
                "path": d["path"],
                "length": len(text),
                "snippet": snippet
            })
        self.logger.info("Ingested %d documents from %s", len(docs), documents_path)
        return summaries

    def list_documents(self):
        """Return a list of document paths currently ingested."""
        return [d["path"] for d in self.documents]

    def get_document_text(self, path: str):
        for d in self.documents:
            if d["path"] == path:
                return d["text"]
        return None
