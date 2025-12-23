function nowMs() { return Date.now(); }
function clamp(n: number, a = -1e9, b = 1e9) { return Math.max(a, Math.min(b, n)); }
function mean(arr: number[]) {
