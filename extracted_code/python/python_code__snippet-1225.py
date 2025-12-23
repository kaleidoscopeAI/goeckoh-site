    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except Exception as exc:  # catch OSError/ImportError from missing shared libs
    TORCH_AVAILABLE = False
    print(f"⚠️  Torch unavailable ({exc}); running without torch/torchaudio")

