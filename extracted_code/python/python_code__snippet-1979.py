"""
Sanitize the "filename" value from a Content-Disposition header.
"""
return os.path.basename(filename)


