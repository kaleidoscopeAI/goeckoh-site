import tempfile
from pathlib import Path
from document_loader import read_all_documents
import pytest

try:
    from reportlab.pdfgen import canvas  # used to create a test PDF
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

def _create_pdf(path: Path, text: str):
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path))
    c.drawString(100, 800, text)
    c.save()

def test_read_txt_and_md(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    t1 = docs_dir / "sample.txt"
    m1 = docs_dir / "sample.md"
    t1.write_text("this is a txt file")
    m1.write_text("# Title\n\nthis is a md file")
    docs = read_all_documents(str(docs_dir))
    paths = {Path(d["path"]).name: d["text"] for d in docs}
    assert "sample.txt" in paths
    assert "sample.md" in paths
    assert "this is a txt file" in paths["sample.txt"]
    assert "this is a md file" in paths["sample.md"]

@pytest.mark.skipif(not HAS_REPORTLAB, reason="reportlab not installed")
def test_read_pdf(tmp_path):
    docs_dir = tmp_path / "docs_pdf"
    docs_dir.mkdir()
    pdf = docs_dir / "sample.pdf"
    text = "Hello PDF"
    _create_pdf(pdf, text)
    docs = read_all_documents(str(docs_dir))
    assert any("sample.pdf" in d["path"] and "Hello PDF" in d["text"] for d in docs)

def test_read_nested_dirs(tmp_path):
    docs_dir = tmp_path / "docs_nested"
    docs_dir.mkdir()
    sub = docs_dir / "subdir"
    sub.mkdir()
    t1 = docs_dir / "top.txt"
    t2 = sub / "nested.txt"
    t1.write_text("top-level file")
    t2.write_text("nested file")
    docs = read_all_documents(str(docs_dir), recursive=True)
    paths = {Path(d["path"]).name: d["text"] for d in docs}
    assert "top.txt" in paths
    assert "nested.txt" in paths
    assert "top-level file" in paths["top.txt"]
    assert "nested file" in paths["nested.txt"]

def test_read_python_files(tmp_path):
    docs_dir = tmp_path / "code"
    docs_dir.mkdir()
    sub = docs_dir / "sub"
    sub.mkdir()
    p1 = docs_dir / "a.py"
    p2 = sub / "b.py"
    p1.write_text("def foo():\n    return 1\n")
    p2.write_text("class C:\n    pass\n")
    from document_loader import read_all_documents
    docs = read_all_documents(str(docs_dir), extensions=[".py"], recursive=True)
    names = {Path(d["path"]).name: d["text"] for d in docs}
    assert "a.py" in names
    assert "b.py" in names
    assert "def foo" in names["a.py"]
    assert "class C" in names["b.py"]
