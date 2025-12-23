def __init__(self) -> None:
    super().__init__()
    self._char_to_freq_order = EUCKR_CHAR_TO_FREQ_ORDER
    self._table_size = EUCKR_TABLE_SIZE
    self.typical_distribution_ratio = EUCKR_TYPICAL_DISTRIBUTION_RATIO

def get_order(self, byte_str: Union[bytes, bytearray]) -> int:
    # for euc-KR encoding, we are interested
    #   first  byte range: 0xb0 -- 0xfe
    #   second byte range: 0xa1 -- 0xfe
    # no validation needed here. State machine has done that
    first_char = byte_str[0]
    if first_char >= 0xB0:
        return 94 * (first_char - 0xB0) + byte_str[1] - 0xA1
    return -1


