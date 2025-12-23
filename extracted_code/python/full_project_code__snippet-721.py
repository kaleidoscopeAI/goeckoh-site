def __init__(self) -> None:
    super().__init__()
    self._char_to_freq_order = JIS_CHAR_TO_FREQ_ORDER
    self._table_size = JIS_TABLE_SIZE
    self.typical_distribution_ratio = JIS_TYPICAL_DISTRIBUTION_RATIO

def get_order(self, byte_str: Union[bytes, bytearray]) -> int:
    # for euc-JP encoding, we are interested
    #   first  byte range: 0xa0 -- 0xfe
    #   second byte range: 0xa1 -- 0xfe
    # no validation needed here. State machine has done that
    char = byte_str[0]
    if char >= 0xA0:
        return 94 * (char - 0xA1) + byte_str[1] - 0xA1
    return -1


