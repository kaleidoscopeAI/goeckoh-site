def __init__(self) -> None:
    super().__init__()
    self._char_to_freq_order = EUCTW_CHAR_TO_FREQ_ORDER
    self._table_size = EUCTW_TABLE_SIZE
    self.typical_distribution_ratio = EUCTW_TYPICAL_DISTRIBUTION_RATIO

def get_order(self, byte_str: Union[bytes, bytearray]) -> int:
    # for euc-TW encoding, we are interested
    #   first  byte range: 0xc4 -- 0xfe
    #   second byte range: 0xa1 -- 0xfe
    # no validation needed here. State machine has done that
    first_char = byte_str[0]
    if first_char >= 0xC4:
        return 94 * (first_char - 0xC4) + byte_str[1] - 0xA1
    return -1


