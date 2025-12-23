"""Get an ideal filename from the given HTTP response, falling back to
the link filename if not provided.
"""
filename = link.filename  # fallback
# Have a look at the Content-Disposition header for a better guess
content_disposition = resp.headers.get("content-disposition")
if content_disposition:
    filename = parse_content_disposition(content_disposition, filename)
ext: Optional[str] = splitext(filename)[1]
if not ext:
    ext = mimetypes.guess_extension(resp.headers.get("content-type", ""))
    if ext:
        filename += ext
if not ext and link.url != resp.url:
    ext = os.path.splitext(resp.url)[1]
    if ext:
        filename += ext
return filename


