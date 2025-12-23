def __init__(self) -> None:
    super().__init__()
    self._char_to_freq_order = JIS_CHAR_TO_FREQ_ORDER
    self._table_size = JIS_TABLE_SIZE
    self.typical_distribution_ratio = JIS_TYPICAL_DISTRIBUTION_RATIO

def get_order(self, byte_str: Union[bytes, bytearray]) -> int:
    # for sjis encoding, we are interested
    #   first  byte range: 0x81 -- 0x9f , 0xe0 -- 0xfe
    #   second byte range: 0x40 -- 0x7e,  0x81 -- oxfe
    # no validation needed here. State machine has done that
    first_char, second_char = byte_str[0], byte_str[1]
    if 0x81 <= first_char <= 0x9F:
        order = 188 * (first_char - 0x81)
    elif 0xE0 <= first_char <= 0xEF:
        order = 188 * (first_char - 0xE0 + 31)
    else:
        return -1
    order = order + second_char - 0x40
    if second_char > 0x7F:
        order = -1
    return order


