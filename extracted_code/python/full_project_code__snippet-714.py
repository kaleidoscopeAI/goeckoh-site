ENOUGH_DATA_THRESHOLD = 1024
SURE_YES = 0.99
SURE_NO = 0.01
MINIMUM_DATA_THRESHOLD = 3

def __init__(self) -> None:
    # Mapping table to get frequency order from char order (get from
    # GetOrder())
    self._char_to_freq_order: Tuple[int, ...] = tuple()
    self._table_size = 0  # Size of above table
    # This is a constant value which varies from language to language,
    # used in calculating confidence.  See
    # http://www.mozilla.org/projects/intl/UniversalCharsetDetection.html
    # for further detail.
    self.typical_distribution_ratio = 0.0
    self._done = False
    self._total_chars = 0
    self._freq_chars = 0
    self.reset()

def reset(self) -> None:
    """reset analyser, clear any state"""
    # If this flag is set to True, detection is done and conclusion has
    # been made
    self._done = False
    self._total_chars = 0  # Total characters encountered
    # The number of characters whose frequency order is less than 512
    self._freq_chars = 0

def feed(self, char: Union[bytes, bytearray], char_len: int) -> None:
    """feed a character with known length"""
    if char_len == 2:
        # we only care about 2-bytes character in our distribution analysis
        order = self.get_order(char)
    else:
        order = -1
    if order >= 0:
        self._total_chars += 1
        # order is valid
        if order < self._table_size:
            if 512 > self._char_to_freq_order[order]:
                self._freq_chars += 1

def get_confidence(self) -> float:
    """return confidence based on existing data"""
    # if we didn't receive any character in our consideration range,
    # return negative answer
    if self._total_chars <= 0 or self._freq_chars <= self.MINIMUM_DATA_THRESHOLD:
        return self.SURE_NO

    if self._total_chars != self._freq_chars:
        r = self._freq_chars / (
            (self._total_chars - self._freq_chars) * self.typical_distribution_ratio
        )
        if r < self.SURE_YES:
            return r

    # normalize confidence (we don't want to be 100% sure)
    return self.SURE_YES

def got_enough_data(self) -> bool:
    # It is not necessary to receive all data to draw conclusion.
    # For charset detection, certain amount of data is enough
    return self._total_chars > self.ENOUGH_DATA_THRESHOLD

def get_order(self, _: Union[bytes, bytearray]) -> int:
    # We do not handle characters based on the original encoding string,
    # but convert this encoding string to a number, here called order.
    # This allows multiple encodings of a language to share one frequency
    # table.
    return -1


