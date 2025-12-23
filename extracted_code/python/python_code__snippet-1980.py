"""
Parse the "filename" value from a Content-Disposition header, and
return the default filename if the result is empty.
"""
m = email.message.Message()
m["content-type"] = content_disposition
filename = m.get_param("filename")
if filename:
    # We need to sanitize the filename to prevent directory traversal
    # in case the filename contains ".." path parts.
    filename = sanitize_content_filename(str(filename))
return filename or default_filename


