def __init__(self) -> None:
    super().__init__()
    self._char_to_freq_order = BIG5_CHAR_TO_FREQ_ORDER
    self._table_size = BIG5_TABLE_SIZE
    self.typical_distribution_ratio = BIG5_TYPICAL_DISTRIBUTION_RATIO

def get_order(self, byte_str: Union[bytes, bytearray]) -> int:
    # for big5 encoding, we are interested
    #   first  byte range: 0xa4 -- 0xfe
    #   second byte range: 0x40 -- 0x7e , 0xa1 -- 0xfe
    # no validation needed here. State machine has done that
    first_char, second_char = byte_str[0], byte_str[1]
    if first_char >= 0xA4:
        if second_char >= 0xA1:
            return 157 * (first_char - 0xA4) + second_char - 0xA1 + 63
        return 157 * (first_char - 0xA4) + second_char - 0x40
    return -1


