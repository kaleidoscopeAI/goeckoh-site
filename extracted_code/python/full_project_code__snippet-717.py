def __init__(self) -> None:
    super().__init__()
    self._char_to_freq_order = EUCKR_CHAR_TO_FREQ_ORDER
    self._table_size = EUCKR_TABLE_SIZE
    self.typical_distribution_ratio = EUCKR_TYPICAL_DISTRIBUTION_RATIO

def get_order(self, byte_str: Union[bytes, bytearray]) -> int:
    first_char = byte_str[0]
    if 0x88 <= first_char < 0xD4:
        code = first_char * 256 + byte_str[1]
        return JOHAB_TO_EUCKR_ORDER_TABLE.get(code, -1)
    return -1


