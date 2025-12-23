import re
from typing import Dict, Any, List
import numpy as np

def validate_data_chunk(data: Dict[str, Any]) -> bool:
    """
    Validates the format and content of a data chunk.
    """
    if not isinstance(data, dict):
        return False

    if "data_type" not in data or "data" not in data:
        raise TypeError("Data chunk does not have 'data_type' or 'data' keys")

    if data["data_type"] == "text":
        return isinstance(data["data"], str) and len(data["data"]) > 0

    elif data["data_type"] == "image":
        # Basic check: you can add more sophisticated image validation here
        return isinstance(data["data"], np.ndarray) and data["data"].ndim == 3

    elif data["data_type"] == "numerical":
        return isinstance(data["data"], (list, np.ndarray))

    else:
        return False  # Unsupported data type

def validate_processing_output(output: Dict[str, Any]) -> bool:
    """
    Validates the format and content of the processing output.
    """
    if not isinstance(output, dict):
        return False

    # Add more specific checks based on your expected output format
    if "processed_data" not in output or "metadata" not in output:
        return False

    return True

