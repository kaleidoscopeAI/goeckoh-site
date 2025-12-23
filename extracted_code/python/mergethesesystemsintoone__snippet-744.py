def __init__(self, allowed_types: List[str] = None, redundancy_threshold: float = 0.9, quality_threshold: float = 0.5):
    """
    Initializes the Membrane with filtering criteria.

    Args:
        allowed_types (List[str]): List of data types allowed through the membrane.
        redundancy_threshold (float): Threshold for filtering redundant data.
        quality_threshold (float): Minimum quality score for data to pass through.
    """
    self.allowed_types = allowed_types if allowed_types is not None else ["text", "numerical", "image", "audio"]
    self.redundancy_threshold = redundancy_threshold
    self.quality_threshold = quality_threshold
    self.memory = []  # Simple memory to track recent data for redundancy checks

def filter_data(self, standardized_data: StandardizedData) -> bool:
    """
    Filters incoming data based on type, quality, and redundancy.

    Args:
        standardized_data (StandardizedData): A StandardizedData object representing the data entry.

    Returns:
        bool: True if the data passes the filter, False otherwise.
    """
    if not self._is_allowed_type(standardized_data):
        logging.info(f"Data type '{standardized_data.data_type}' is not allowed.")
        return False

    if not self._meets_quality_threshold(standardized_data):
        logging.info(f"Data quality score '{standardized_data.quality_score}' is below the threshold.")
        return False

    if self._is_redundant(standardized_data):
        logging.info(f"Data entry is redundant: '{standardized_data.content_summary}'")
        return False

    self.memory.append(standardized_data)
    if len(self.memory) > 100:  # Limit memory size
        self.memory.pop(0)
    return True

def _is_allowed_type(self, standardized_data: StandardizedData) -> bool:
    """
    Checks if the data type is allowed.

    Args:
        standardized_data (StandardizedData): The standardized data object.

    Returns:
        bool: True if the data type is allowed, False otherwise.
    """
    return standardized_data.data_type in self.allowed_types

def _meets_quality_threshold(self, standardized_data: StandardizedData) -> bool:
    """
    Checks if the data meets the minimum quality threshold.

    Args:
        standardized_data (StandardizedData): The standardized data object.

    Returns:
        bool: True if the quality score meets the threshold, False otherwise.
    """
    return standardized_data.quality_score >= self.quality_threshold

def _is_redundant(self, standardized_data: StandardizedData) -> bool:
    """
    Checks if the data is redundant based on recent data.
    Uses a simple similarity check based on metadata and content summary.
    """
    new_content_summary = standardized_data.content_summary
    for existing_data in self.memory:
        if self._calculate_similarity(new_content_summary, existing_data.content_summary) > self.redundancy_threshold:
            return True
    return False

def _calculate_similarity(self, summary1: str, summary2: str) -> float:
    """
    Calculates a simple similarity score between two content summaries.

    Args:
        summary1 (str): The first content summary.
        summary2 (str): The second content summary.

    Returns:
        float: A similarity score between 0.0 and 1.0.
    """
    # Convert to lowercase for case-insensitive comparison
    summary1 = summary1.lower()
    summary2 = summary2.lower()

    # Calculate similarity ratio
    longer_length = max(len(summary1), len(summary2))
    if longer_length == 0:
        return 1.0  # Both strings are empty

    common_chars = 0
    for char in summary1:
        if char in summary2:
            common_chars += 1

    similarity = common_chars / longer_length
    return similarity

def get_state(self) -> dict:
    """Returns the current state of the membrane."""
    return {
        'allowed_types': self.allowed_types,
        'redundancy_threshold': self.redundancy_threshold,
        'quality_threshold': self.quality_threshold,
        'memory_size': len(self.memory)
    }


