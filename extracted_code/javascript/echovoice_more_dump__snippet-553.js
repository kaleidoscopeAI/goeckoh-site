function handlePointerChange(idx: number, newVal: number) {
const e = emotions[idx];
const patch: Partial<EVector> = { [e]: Math.max(-1, Math.min(1, newVal)) } as any;
