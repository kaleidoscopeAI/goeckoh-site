function entropy(data: number[]): number {
if(!data.length) return 0.0; const sum = data.reduce((a,d)=>a+Math.abs(d),0); if(sum<=0.0) return 0.0;
return data.reduce((r,d)=>{ const p=Math.abs(d)/sum; return p>0.0 ? r - pMath.log(p) : r; }, 0.0);
