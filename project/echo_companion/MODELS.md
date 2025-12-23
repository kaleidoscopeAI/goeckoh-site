Model locations (RTVC)
======================

Real-Time-Voice-Cloning downloads three pretrained checkpoints when you run `setup.sh`:
- encoder/saved_models/pretrained.pt
- synthesizer/saved_models/pretrained/pretrained.pt
- vocoder/saved_models/pretrained/pretrained.pt

They live under:
  echo_companion/third_party/Real-Time-Voice-Cloning/

If you want to keep a fully offline bundle, you can pre-place those files in that
directory or archive the repo with models yourself; otherwise `setup.sh` will
clone RTVC and run its downloader once (network required for that step only).
