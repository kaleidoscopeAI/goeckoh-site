    On startup do this, no UI needed:
         Start the loop (run() task) automatically.
         Schedule ingest every AUTONOMOUS_INGEST_EVERY ticks from three sources in priority:

          1. Local corpus (project files, previously ingested docs).
          2. Offline caches (anything under seed_crystal_agi/state/corpus/).
          3. Optional web crawl (only if ALLOW_ONLINE=1).

    If Ollama not reachable, reflection uses deterministic text (we already have make_reflection() + heuristic adjust).
    Never block on missing models; always fall back.

