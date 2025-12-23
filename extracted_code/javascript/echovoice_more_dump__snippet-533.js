private _sanitizeAndClampVector(input: Partial<EVector>): EVector {
const out: any = makeZeroVector(this.emotions);
for (const k of this.emotions) {
const v = (input as any)[k];
