let mut state: u64 = 0;
state |= (cpu_ms.min(BIT_MASK_13) & BIT_MASK_13) << 0;
state |= (mem_kb.min(BIT_MASK_13) & BIT_MASK_13) << 13;
state |= (io_total.min(BIT_MASK_13) & BIT_MASK_13) << 26;
state |= (((prio + 20) as u64).min(BIT_MASK_13) & BIT_MASK_13) << 39;  // Nice -20..19 -> 0..39
state |= (entropy & BIT_MASK_12) << 52;
state
