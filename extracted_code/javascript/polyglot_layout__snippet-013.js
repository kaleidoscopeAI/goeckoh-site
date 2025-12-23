function integratedInformation(vec: number[]): number {
const n=vec.length; if(!n) return 0.0; const parts=Math.max(1, Math.floor(n/2));
const sys = entropy(vec); let part=0.0;
for(let i=0;i<parts;i++){ const sub:number=[]; for(let j=i;j<n;j+=parts) sub.push(vec[j]); part += entropy(sub); }
