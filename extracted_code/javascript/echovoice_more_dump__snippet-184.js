fn unpack_state(state: u64) -> (u64, u64, u64, i32, u64) {
    let cpu = (state >> 0) & BIT_MASK_13;
    let mem = (state >> 13) & BIT_MASK_13;
    let io = (state >> 26) & BIT_MASK_13;
    let prio = (((state >> 39) & BIT_MASK_13) as i32) - 20;
    let entropy = (state >> 52) & BIT_MASK_12;
    (cpu, mem, io, prio, entropy)
