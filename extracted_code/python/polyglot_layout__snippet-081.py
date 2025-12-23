  p = Path(path)
  if not p.exists(): return JSONResponse({"ok":False,"error":"not found"}, status_code=404)
  from fastapi.responses import FileResponse
  return FileResponse(str(p))

