import platform
import multiprocessing
from dataclasses import dataclass

import torch


@dataclass
class HardwareSummary:
    os: str
    cpu: str
    cores: int
    has_cuda: bool
    cuda_device: str
    suggested_whisper_model: str


def get_hardware_summary() -> HardwareSummary:
    os_name = f"{platform.system()} {platform.release()}"
    cpu_name = platform.processor() or "Unknown CPU"
    cores = multiprocessing.cpu_count()

    has_cuda = torch.cuda.is_available()
    cuda_device = torch.cuda.get_device_name(0) if has_cuda else "None"

    # Crude heuristic for model size
    if has_cuda:
        # GPU present: "small" is a safe starting point
        suggested = "small"
    else:
        # CPU only
        if cores <= 4:
            suggested = "tiny"
        else:
            suggested = "base"

    return HardwareSummary(
        os=os_name,
        cpu=cpu_name,
        cores=cores,
        has_cuda=has_cuda,
        cuda_device=cuda_device,
        suggested_whisper_model=suggested,
    )
