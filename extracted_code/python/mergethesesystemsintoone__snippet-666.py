def __init__(self):
    super().__init__("numerical")

def process(self, data_wrapper: DataWrapper) -> np.ndarray:
    """Processes numerical data. Returns a standardized numerical array."""
    data = data_wrapper.get_data()

    if isinstance(data, list):
        data = np.array(data)

    if data.ndim != 1:
        raise ValueError("Numerical data must be a 1D array or list")

    mean = np.mean(data)
    std = np.std(data)
    standardized_data = (data - mean) / std if std != 0 else np.zeros_like(data)

    return standardized_data

