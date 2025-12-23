def pack_bits(bit_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(bit_array).astype(np.uint8).ravel()
    if arr.size == 0:
        return np.zeros(0, dtype=np.uint64)
    pad = (-arr.size) % _WORD_BITS
    if pad > 0:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    arr_bits = arr.reshape(-1, _WORD_BITS)
    # packbits returns bytes (per row) in big-endian or little; we use little
    bytes_rows = np.packbits(arr_bits, axis=1, bitorder='little')
    words = np.frombuffer(bytes_rows.tobytes(), dtype=np.uint64)
    return words

def popcount_u64(arr: np.ndarray) -> int:
    if arr.size == 0:
        return 0
    # efficient path: use vectorized popcount via numpy's unpackbits on uint8 view
    bytes_view = arr.view(np.uint8)
    bits = np.unpackbits(bytes_view, bitorder='little')
    return int(bits.sum())

def popcount_xor(a_packed: np.ndarray, b_packed: np.ndarray) -> int:
    n = max(a_packed.size, b_packed.size)
    if a_packed.size != n:
        a = np.pad(a_packed, (0, n - a_packed.size), constant_values=0)
    else:
        a = a_packed
    if b_packed.size != n:
        b = np.pad(b_packed, (0, n - b_packed.size), constant_values=0)
    else:
        b = b_packed
    xor = np.bitwise_xor(a, b)
    return popcount_u64(xor)

