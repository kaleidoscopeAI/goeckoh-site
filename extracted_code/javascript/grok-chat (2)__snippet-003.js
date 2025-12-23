  stateEmb(s) { const e=Array(32).fill(0).map((*,i)=>((s>>>i)&1)*2-1); const n=Math.sqrt(e.reduce((a,b)=>a+b*b,0))+.001; return e.map(v=>v/n); }
  cosSim(a,b) { let d=0,ma=0,mb=0; for(let i=0;i<a.length;i++){d+=a[i]*b[i];ma+=a[i]*a[i];mb+=b[i]*b[i];} return d/(Math.sqrt(ma*mb)+.0001); }
  interpret(s) { const e=this.stateEmb(s); return CONCEPTS.map(c=>({c,s:this.cosSim(e,this.emb[c])})).sort((a,b)=>b.s-a.s).slice(0,3); }
