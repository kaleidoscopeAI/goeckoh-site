const mwRaw = projectVec(s, DEFAULT_P.memory_write);
const memoryWrite = clamp(1 + mwRaw, 0, DEFAULT_PARAMS.W_MAX);
const memoryTagLifetimeMult = clamp(1 + 0.5 * mwRaw, 0.1, 10);
