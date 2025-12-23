  separately from playback duration by having _speak return both, then storing
  processing latency and playback latency distinctly; this will improve accuracy
  of latency metrics without a large refactor. Also preparing to update
  reference audio loading to float32 with resampling.

