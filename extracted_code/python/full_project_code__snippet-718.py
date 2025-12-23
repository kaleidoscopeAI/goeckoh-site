def __init__(self) -> None:
    super().__init__()
    self._char_to_freq_order = GB2312_CHAR_TO_FREQ_ORDER
    self._table_size = GB2312_TABLE_SIZE
    self.typical_distribution_ratio = GB2312_TYPICAL_DISTRIBUTION_RATIO

def get_order(self, byte_str: Union[bytes, bytearray]) -> int:
    # for GB2312 encoding, we are interested
    #  first  byte range: 0xb0 -- 0xfe
    #  second byte range: 0xa1 -- 0xfe
    # no validation needed here. State machine has done that
    first_char, second_char = byte_str[0], byte_str[1]
    if (first_char >= 0xB0) and (second_char >= 0xA1):
        return 94 * (first_char - 0xB0) + second_char - 0xA1
    return -1


