for (const t of logs) {
const weight = (t.regret ?? 0) / totalRegret;
for (const k in t.mods.raw) {
for (let i = 0; i < t.mods.raw[k].length; i++) {
