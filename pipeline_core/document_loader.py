from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger("document_loader")

def _read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()

def _read_pdf(path: Path) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception as e:
        raise RuntimeError("PyPDF2 is required for reading PDF files.") from e
    try:
        reader = PdfReader(str(path))
        pages = []
        for p in reader.pages:
            text = p.extract_text() or ""
            pages.append(text)
        return "\n".join(pages)
    except Exception as e:
        logger.exception("Failed to extract PDF text from %s", path)
        raise RuntimeError(f"Failed to extract PDF text: {e}") from e

def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".txt", ".md", ".py"]:
        # Read code or plain text files as raw text
        return _read_text_file(path)
    elif suffix == ".pdf":
        return _read_pdf(path)
    else:
        raise RuntimeError(f"Unsupported file type: {path}")

def read_all_documents(folder: str, extensions: List[str] = None, recursive: bool = True) -> List[Dict[str, str]]:
    """
    Read all documents from 'folder'. If recursive is True, traverse subfolders.
    Returns list of dicts: {"path": path_str, "text": content}
    """
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        raise RuntimeError(f"Documents folder does not exist or is not a directory: {folder}")
    # Default to text, markdown and pdfs unless explicitly provided.
    extensions = extensions or [".txt", ".md", ".pdf"]
    documents = []
    if recursive:
        iterator = p.rglob("*")
    else:
        iterator = p.iterdir()
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        try:
            content = read_document(path)
            documents.append({"path": str(path), "text": content})
        except Exception as e:
            logger.warning("Skipping %s due to error: %s", path, e)
    return documents
