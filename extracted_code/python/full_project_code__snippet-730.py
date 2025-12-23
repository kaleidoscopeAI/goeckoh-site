def get_order(self, byte_str: Union[bytes, bytearray]) -> Tuple[int, int]:
    if not byte_str:
        return -1, 1
    # find out current char's byte length
    first_char = byte_str[0]
    if (first_char == 0x8E) or (0xA1 <= first_char <= 0xFE):
        char_len = 2
    elif first_char == 0x8F:
        char_len = 3
    else:
        char_len = 1

    # return its order if it is hiragana
    if len(byte_str) > 1:
        second_char = byte_str[1]
        if (first_char == 0xA4) and (0xA1 <= second_char <= 0xF3):
            return second_char - 0xA1, char_len

    return -1, char_len


