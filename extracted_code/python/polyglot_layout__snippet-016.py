  import os, zipfile, io, sys, re, json, textwrap
  from pathlib import Path
  from typing import List, Dict, Tuple
  import pandas as pd
  from collections import Counter, defaultdict
  from caas_jupyter_tools import display_dataframe_to_user

  BASE = Path("/mnt/data")
  ZIP_A = BASE / "organic_ai_full_infra_bundle.zip"
  ZIP_B = BASE / "cognitive-nebula(1).zip"
  OUT_DIR = BASE / "extracted"
  OUT_DIR.mkdir(parents=True, exist_ok=True)

  def safe_unzip(zip_path: Path, dest_dir: Path) -> Tuple[bool, str, int]:
      if not zip_path.exists():
           return False, f"Missing: {zip_path}", 0
      try:
           with zipfile.ZipFile(zip_path, 'r') as z:
               # prevent zip-slip
               count = 0
               for m in z.infolist():
                   # normalize target
                   target = dest_dir / m.filename
                   # resolve and ensure within dest_dir
                   if not str(target.resolve()).startswith(str(dest_dir.resolve())):
                       continue # skip suspicious paths
                   if m.is_dir():
                       target.mkdir(parents=True, exist_ok=True)
                   else:
                       target.parent.mkdir(parents=True, exist_ok=True)
                       with z.open(m) as src, open(target, 'wb') as dst:
                           dst.write(src.read())
                   count += 1
           return True, "ok", count
      except Exception as e:
           return False, repr(e), 0

  E_A = OUT_DIR / "organic_ai_full_infra_bundle"

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                                                       78/124

  E_B = OUT_DIR / "cognitive-nebula_1"
  E_A.mkdir(exist_ok=True, parents=True)
  E_B.mkdir(exist_ok=True, parents=True)

  status = []
  status.append(("organic_ai_full_infra_bundle.zip",) + safe_unzip(ZIP_A, E_A))
  status.append(("cognitive-nebula(1).zip",) + safe_unzip(ZIP_B, E_B))

  status_df = pd.DataFrame(status, columns=["archive", "ok", "msg", "files_extracted"])

  # Walk both and collect manifest (limit to 5000 files to stay responsive)
  def collect_manifest(root: Path, cap=5000) -> List[Dict]:
      rows = []
      for i, p in enumerate(root.rglob("*")):
          if i >= cap:
              break
          if p.is_file():
              try:
                   size = p.stat().st_size
              except Exception:
                   size = None
              rel = p.relative_to(root).as_posix()
              rows.append({
                   "root": root.name,
                   "relpath": rel,
                   "name": p.name,
                   "ext": p.suffix.lower(),
                   "size_bytes": size
              })
      return rows

  manifest_rows = collect_manifest(E_A, cap=15000) + collect_manifest(E_B, cap=15000)
  manifest_df = pd.DataFrame(manifest_rows)

  # Quick language/ext aggregation
  ext_alias = {
      ".py":"Python",".ipynb":"Jupyter",".js":"JavaScript",".ts":"TypeScript",".tsx":"TypeScript JSX",
      ".jsx":"React JSX",".json":"JSON",".md":"Markdown",".txt":"Text",".sql":"SQL",".yml":"YAML",".yaml":"YAML",
      ".toml":"TOML",".ini":"INI",".cfg":"Config",".sh":"Shell",".bat":"Batch",".ps1":"PowerShell",
      ".html":"HTML",".css":"CSS",".scss":"SCSS",".less":"LESS",".java":"Java",".kt":"Kotlin",".kts":"Kotlin",
      ".c":"C",".cpp":"C++",".h":"C/C++ Header",".hpp":"C++ Header",".rs":"Rust",".go":"Go",".rb":"Ruby",
      ".php":"PHP",".swift":"Swift",".mm":"Obj-C++",".m":"Obj-C/C Matlab",".glsl":"GLSL",".vert":"GLSL",
      ".frag":"GLSL",".wgsl":"WGSL",".cl":"OpenCL",".cu":"CUDA",".proto":"Protobuf",".pb":"Protobuf",
      ".s":"Assembly",".asm":"Assembly",".pyx":"Cython",".pxd":"Cython",".gradle":"Gradle",
      ".cs":"C#",".fs":"F#",".r":"R",".mjs":"JavaScript",".cjs":"JavaScript",".tsv":"TSV",".csv":"CSV",
  }
  manifest_df["type"] = manifest_df["ext"].map(lambda e: ext_alias.get(e, e if e else "noext"))

  type_counts = manifest_df.groupby(["root","type"]).size().reset_index(name="count").sort_values(["root","count"], ascending=[True, False])
  top_levels = []
  for root in [E_A, E_B]:
      # list immediate children
      items = []
      for p in sorted(root.glob("*")):
          items.append((root.name, p.name + ("/" if p.is_dir() else ""), "dir" if p.is_dir() else "file"))
      top_levels.extend(items)
  top_df = pd.DataFrame(top_levels, columns=["root","item","kind"])

  # Find key config files
  KEY_NAMES = {"readme.md","readme","requirements.txt","package.json","pyproject.toml",
               "setup.py","environment.yml","Dockerfile","docker-compose.yml","build.gradle","build.gradle.kts"}
  key_files = manifest_df[manifest_df["name"].str.lower().isin(KEY_NAMES)].copy()

  # Search signals inside text files (limited to 1MB per file and up to 400 hits total)
  signals = [
      (r"\b18[ ,]?000\b", "mentions_18000"),
      (r"\bavatar\b", "avatar"),
      (r"\bnodes?\b", "nodes"),
      (r"\bthree\.?js\b", "threejs"),
      (r"\bwebgl\b", "webgl"),
      (r"\bglsl\b", "glsl"),
      (r"\bshader\b", "shader"),
      (r"\bopen(gl| gles)\b", "opengl"),
      (r"\bollama\b", "ollama"),
      (r"\bfastapi\b", "fastapi"),
      (r"\buvicorn\b", "uvicorn"),
      (r"\bkotlin\b", "kotlin"),
      (r"\bandroid\b", "android"),
      (r"\bnode[s ]*count\b", "node_count"),
      (r"\b18000\s*nodes\b", "exact_18000_nodes"),
  ]
  signal_results = []
  hits_limit = 400

  def is_textual(path: Path) -> bool:
      # Heuristic based on extension
      if path.suffix.lower() in [".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".ico",
                                 ".mp3",".wav",".flac",".ogg",".zip",".tar",".gz",".xz",".7z",
                                 ".pdf",".parquet",".feather",".npy",".npz",".pt",".bin",".onnx",".tflite"]:
          return False
      return True

  for _, row in manifest_df.iterrows():
      if len(signal_results) >= hits_limit:
           break
      root = E_A if row["root"] == E_A.name else E_B
      p = root / row["relpath"]
      try:
           if not p.is_file() or not is_textual(p) or (p.stat().st_size or 0) > 1_000_000:
               continue
           txt = p.read_text(encoding="utf-8", errors="ignore")
      except Exception:
           continue
      for pattern, tag in signals:
           for m in re.finditer(pattern, txt, flags=re.IGNORECASE):
               start = max(0, m.start()-80)
               end = min(len(txt), m.end()+80)
               snippet = txt[start:end].replace("\n"," ")
               signal_results.append({
                   "root": row["root"],

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                                             79/124

                   "relpath": row["relpath"],
                   "signal": tag,
                   "pattern": pattern,
                   "snippet": snippet
               })
               if len(signal_results) >= hits_limit:
                   break
           if len(signal_results) >= hits_limit:
               break

  signals_df = pd.DataFrame(signal_results)

  # Try to read top README-like files for quick context (first 120 lines)
  readme_snippets = []
  for _, r in key_files.iterrows():
      root = E_A if r["root"] == E_A.name else E_B
      p = root / r["relpath"]
      if p.name.lower().startswith("readme") and p.is_file():
          try:
              lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()[:120]
              readme_snippets.append({
                   "root": r["root"],
                   "relpath": r["relpath"],
                   "snippet": "\n".join(lines)
              })
          except Exception:
              pass
  readme_df = pd.DataFrame(readme_snippets)

  # Save a JSON manifest + concise markdown summary
  summary_dir = BASE / "zip_review"
  summary_dir.mkdir(parents=True, exist_ok=True)

  manifest_path = summary_dir / "manifest_subset.json"
  with open(manifest_path, "w", encoding="utf-8") as f:
      json.dump(manifest_rows[:5000], f, indent=2)

  summary_md = summary_dir / "summary.md"
  with open(summary_md, "w", encoding="utf-8") as f:
      f.write("# Zip Review Summary\n\n")
      f.write("## Extraction Status\n\n")
      f.write(status_df.to_markdown(index=False))
      f.write("\n\n## Top-level Items\n\n")
      try:
           f.write(top_df.to_markdown(index=False))
      except Exception:
           f.write(top_df.to_csv(index=False))
      f.write("\n\n## File Types (by root)\n\n")
      try:
           f.write(type_counts.to_markdown(index=False))
      except Exception:
           f.write(type_counts.to_csv(index=False))
      f.write("\n\n## Key Config Files\n\n")
      try:
           f.write(key_files.to_markdown(index=False))
      except Exception:
           f.write(key_files.to_csv(index=False))
      f.write("\n\n## Signal Hits (first ~400)\n\n")
      try:
           f.write(signals_df.head(100).to_markdown(index=False))
      except Exception:
           f.write(signals_df.head(100).to_csv(index=False))
      f.write("\n\n## README Snippets (first 120 lines each)\n\n")
      for _, rr in readme_df.iterrows():
           f.write(f"### {rr['root']}/{rr['relpath']}\n\n")
           f.write("```\n")
           f.write(rr["snippet"])
           f.write("\n```\n\n")

  # Show key tables in the UI
  display_dataframe_to_user("Extraction status", status_df)
  display_dataframe_to_user("Top-level items", top_df)
  display_dataframe_to_user("File types by root", type_counts)
  if not key_files.empty:
      display_dataframe_to_user("Key config files detected", key_files)
  if not signals_df.empty:
      display_dataframe_to_user("Signal hits (subset)", signals_df.head(200))
  if not readme_df.empty:
      display_dataframe_to_user("README snippets (where present)", readme_df)

  # Provide paths for download
  (str(manifest_path), str(summary_md))


  python


  # Retry: Unzip and review archives, build manifest, show tables, and save summary files.

  import os, zipfile, re, json
  from pathlib import Path
  import pandas as pd
  from caas_jupyter_tools import display_dataframe_to_user

  BASE = Path("/mnt/data")
  ZIP_A = BASE / "organic_ai_full_infra_bundle.zip"
  ZIP_B = BASE / "cognitive-nebula(1).zip"
  OUT_DIR = BASE / "extracted"
  OUT_DIR.mkdir(parents=True, exist_ok=True)

  def safe_unzip(zip_path: Path, dest_dir: Path):
      if not zip_path.exists():
           return False, f"Missing: {zip_path}", 0
      try:
           with zipfile.ZipFile(zip_path, 'r') as z:
               count = 0
               for m in z.infolist():
                   target = dest_dir / m.filename
                   # prevent zip slip
                   if not str(target.resolve()).startswith(str(dest_dir.resolve())):

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                           80/124

                      continue
                  if m.is_dir():
                      target.mkdir(parents=True, exist_ok=True)
                  else:
                      target.parent.mkdir(parents=True, exist_ok=True)
                      with z.open(m) as src, open(target, 'wb') as dst:
                          dst.write(src.read())
                  count += 1
          return True, "ok", count
      except Exception as e:
          return False, repr(e), 0

  E_A = OUT_DIR / "organic_ai_full_infra_bundle"
  E_B = OUT_DIR / "cognitive-nebula_1"
  E_A.mkdir(exist_ok=True, parents=True)
  E_B.mkdir(exist_ok=True, parents=True)

  status = []
  status.append(("organic_ai_full_infra_bundle.zip",) + safe_unzip(ZIP_A, E_A))
  status.append(("cognitive-nebula(1).zip",) + safe_unzip(ZIP_B, E_B))
  status_df = pd.DataFrame(status, columns=["archive", "ok", "msg", "files_extracted"])

  # Build manifest
  def collect_manifest(root: Path, cap=15000):
      rows = []
      i = 0
      for p in root.rglob("*"):
          if i >= cap:
              break
          if p.is_file():
              rows.append({
                   "root": root.name,
                   "relpath": p.relative_to(root).as_posix(),
                   "name": p.name,
                   "ext": p.suffix.lower(),
                   "size_bytes": p.stat().st_size
              })
              i += 1
      return rows

  manifest_rows = collect_manifest(E_A) + collect_manifest(E_B)
  manifest_df = pd.DataFrame(manifest_rows)

  # Types summary
  ext_alias = {
      ".py":"Python",".ipynb":"Jupyter",".js":"JavaScript",".ts":"TypeScript",".tsx":"TypeScript JSX",
      ".jsx":"React JSX",".json":"JSON",".md":"Markdown",".txt":"Text",".sql":"SQL",".yml":"YAML",".yaml":"YAML",
      ".toml":"TOML",".ini":"INI",".cfg":"Config",".sh":"Shell",".bat":"Batch",".ps1":"PowerShell",
      ".html":"HTML",".css":"CSS",".scss":"SCSS",".less":"LESS",".java":"Java",".kt":"Kotlin",".kts":"Kotlin",
      ".c":"C",".cpp":"C++",".h":"C/C++ Header",".hpp":"C++ Header",".rs":"Rust",".go":"Go",".rb":"Ruby",
      ".php":"PHP",".swift":"Swift",".mm":"Obj-C++",".m":"Obj-C/C Matlab",".glsl":"GLSL",".vert":"GLSL",
      ".frag":"GLSL",".wgsl":"WGSL",".cl":"OpenCL",".cu":"CUDA",".proto":"Protobuf",".s":"Assembly",".asm":"Assembly",
      ".gradle":"Gradle",".r":"R",".mjs":"JavaScript",".cjs":"JavaScript",".csv":"CSV",".tsv":"TSV",
  }
  manifest_df["type"] = manifest_df["ext"].map(lambda e: ext_alias.get(e, e if e else "noext"))
  type_counts = manifest_df.groupby(["root","type"]).size().reset_index(name="count").sort_values(["root","count"], ascending=[True, False])

  # Top-level listing
  top_items = []
  for root in [E_A, E_B]:
      for p in sorted(root.glob("*")):
          top_items.append((root.name, p.name + ("/" if p.is_dir() else ""), "dir" if p.is_dir() else "file"))
  top_df = pd.DataFrame(top_items, columns=["root","item","kind"])

  # Key config files
  KEY_NAMES = {"readme.md","readme","requirements.txt","package.json","pyproject.toml",
               "setup.py","environment.yml","Dockerfile","docker-compose.yml","build.gradle","build.gradle.kts"}
  key_files = manifest_df[manifest_df["name"].str.lower().isin(KEY_NAMES)].copy()

  # Scan for signals
  signals = [
      (r"\b18[ ,]?000\b", "mentions_18000"),
      (r"\bavatar\b", "avatar"),
      (r"\bnodes?\b", "nodes"),
      (r"\bthree\.?js\b", "threejs"),
      (r"\bwebgl\b", "webgl"),
      (r"\bglsl\b", "glsl"),
      (r"\bshader\b", "shader"),
      (r"\bollama\b", "ollama"),
      (r"\bfastapi\b", "fastapi"),
      (r"\buvicorn\b", "uvicorn"),
      (r"\bkotlin\b", "kotlin"),
      (r"\bandroid\b", "android"),
      (r"\b18000\s*nodes\b", "exact_18000_nodes"),
  ]
  signal_rows = []
  hits = 0
  for _, row in manifest_df.iterrows():
      if hits >= 400:
           break
      root = E_A if row["root"] == E_A.name else E_B
      path = root / row["relpath"]
      # Skip binary-ish files by extension
      if row["ext"] in
  [".png",".jpg",".jpeg",".gif",".bmp",".tif",".tiff",".ico",".pdf",".zip",".tar",".gz",".xz",".7z",".mp3",".wav",".ogg",".flac",".pt",".bin",".on
  nx",".tflite",".npy",".npz",".mp4",".mov",".avi",".webm"]:
           continue
      if (path.stat().st_size or 0) > 1_000_000:
           continue
      try:
           txt = path.read_text(encoding="utf-8", errors="ignore")
      except Exception:
           continue
      for pat, tag in signals:
           for m in re.finditer(pat, txt, flags=re.IGNORECASE):
               snippet = txt[max(0, m.start()-80): m.end()+80].replace("\n"," ")
               signal_rows.append({"root": row["root"], "relpath": row["relpath"], "signal": tag, "snippet": snippet})
               hits += 1
               if hits >= 400:
                   break

Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                                             81/124

          if hits >= 400:
              break
  signals_df = pd.DataFrame(signal_rows)

  # Persist a compact manifest + summary
  summary_dir = BASE / "zip_review"
  summary_dir.mkdir(parents=True, exist_ok=True)
  manifest_json = summary_dir / "manifest_subset.json"
  manifest_json.write_text(json.dumps(manifest_rows[:5000], indent=2), encoding="utf-8")

  summary_md = summary_dir / "summary.md"
  with open(summary_md, "w", encoding="utf-8") as f:
      f.write("# Zip Review Summary\n\n")
      f.write(status_df.to_markdown(index=False) + "\n\n")
      f.write("## Top-level Items\n")
      f.write(top_df.to_markdown(index=False) + "\n\n")
      f.write("## File Types by Root\n")
      f.write(type_counts.to_markdown(index=False) + "\n\n")
      f.write("## Key Config Files\n")
      try:
           f.write(key_files.to_markdown(index=False) + "\n\n")
      except:
           pass
      f.write("## Signal Hits (subset)\n")
      try:
           f.write(signals_df.head(100).to_markdown(index=False) + "\n\n")
      except:
           pass

  # Display tables for quick review
  display_dataframe_to_user("Extraction status", status_df)
  display_dataframe_to_user("Top-level items", top_df)
  display_dataframe_to_user("File types by root", type_counts)
  if not key_files.empty:
      display_dataframe_to_user("Key config files detected", key_files)
  if not signals_df.empty:
      display_dataframe_to_user("Signal hits (subset)", signals_df.head(200))

  # Return output file paths for download
  print(json.dumps({
      "manifest_subset": str(manifest_json),
      "summary_markdown": str(summary_md)
  }))


  STDOUT/STDERR
  {"manifest_subset": "/mnt/data/zip_review/manifest_subset.json", "summary_markdown": "/mnt/data/zip_review/summary.md"}

  Extraction status




  Top-level items




  File types by root




Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.                                                          82/124

  Key config files detected




  Signal hits (subset)




