def __init__(self) -> None:
    super().__init__()
    self._charset_name = "SHIFT_JIS"

@property
def charset_name(self) -> str:
    return self._charset_name

def get_order(self, byte_str: Union[bytes, bytearray]) -> Tuple[int, int]:
    if not byte_str:
        return -1, 1
    # find out current char's byte length
    first_char = byte_str[0]
    if (0x81 <= first_char <= 0x9F) or (0xE0 <= first_char <= 0xFC):
        char_len = 2
        if (first_char == 0x87) or (0xFA <= first_char <= 0xFC):
            self._charset_name = "CP932"
    else:
        char_len = 1

    # return its order if it is hiragana
    if len(byte_str) > 1:
        second_char = byte_str[1]
        if (first_char == 202) and (0x9F <= second_char <= 0xF1):
            return second_char - 0x9F, char_len

    return -1, char_len


