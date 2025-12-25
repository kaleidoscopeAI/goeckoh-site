# Documents Directory

Place text documents here for the system to read and process.

## Supported Formats

- `.txt` - Plain text files
- `.md` - Markdown files
- `.pdf` - PDF documents (requires PyPDF2 or similar)

## Usage

The system will automatically read all supported documents in this directory
when configured with:

```yaml
documents_path: "./documents"
```

You can use the CLI to read documents:

```bash
python -m cli read-docs --path ./documents --recursive
```
