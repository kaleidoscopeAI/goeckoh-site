  import glob, html
  frames = sorted(glob.glob(str((OUT_SHAPES / 'avatar_*.png').absolute())))
  latest = frames[-1] if frames else ""; latest_rel = os.path.basename(latest) if latest else ""
  body = f"<h1>Seed-Crystal AGI</h1><p>Access <code>/status</code>, <code>/recent</code>, or <code>/ws</code>.</p>"
  if latest_rel: body += f"<p>Latest avatar frame: <code>{html.escape(latest_rel)}</code></p>"
  return "<html><body>"+body+"</body></html>"

