for (const t of logs) {
const weight = (t.regret ?? 0.0) / totalRegret;
for (const k in t.mods?.raw ?? {}) {
const rawVec: FloatArr = t.mods.raw[k];
for (let i = 0; i < rawVec.length; i++) {
