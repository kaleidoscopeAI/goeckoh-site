NUM_OF_CATEGORY = 6
DONT_KNOW = -1
ENOUGH_REL_THRESHOLD = 100
MAX_REL_THRESHOLD = 1000
MINIMUM_DATA_THRESHOLD = 4

def __init__(self) -> None:
    self._total_rel = 0
    self._rel_sample: List[int] = []
    self._need_to_skip_char_num = 0
    self._last_char_order = -1
    self._done = False
    self.reset()

def reset(self) -> None:
    self._total_rel = 0  # total sequence received
    # category counters, each integer counts sequence in its category
    self._rel_sample = [0] * self.NUM_OF_CATEGORY
    # if last byte in current buffer is not the last byte of a character,
    # we need to know how many bytes to skip in next buffer
    self._need_to_skip_char_num = 0
    self._last_char_order = -1  # The order of previous char
    # If this flag is set to True, detection is done and conclusion has
    # been made
    self._done = False

def feed(self, byte_str: Union[bytes, bytearray], num_bytes: int) -> None:
    if self._done:
        return

    # The buffer we got is byte oriented, and a character may span in more than one
    # buffers. In case the last one or two byte in last buffer is not
    # complete, we record how many byte needed to complete that character
    # and skip these bytes here.  We can choose to record those bytes as
    # well and analyse the character once it is complete, but since a
    # character will not make much difference, by simply skipping
    # this character will simply our logic and improve performance.
    i = self._need_to_skip_char_num
    while i < num_bytes:
        order, char_len = self.get_order(byte_str[i : i + 2])
        i += char_len
        if i > num_bytes:
            self._need_to_skip_char_num = i - num_bytes
            self._last_char_order = -1
        else:
            if (order != -1) and (self._last_char_order != -1):
                self._total_rel += 1
                if self._total_rel > self.MAX_REL_THRESHOLD:
                    self._done = True
                    break
                self._rel_sample[
                    jp2_char_context[self._last_char_order][order]
                ] += 1
            self._last_char_order = order

def got_enough_data(self) -> bool:
    return self._total_rel > self.ENOUGH_REL_THRESHOLD

def get_confidence(self) -> float:
    # This is just one way to calculate confidence. It works well for me.
    if self._total_rel > self.MINIMUM_DATA_THRESHOLD:
        return (self._total_rel - self._rel_sample[0]) / self._total_rel
    return self.DONT_KNOW

def get_order(self, _: Union[bytes, bytearray]) -> Tuple[int, int]:
    return -1, 1


