  const memF = s => { let m=1; for(let i=0;i<32;i++) m*=1-.3*((((s>>>i)&1)-.5)/.5)**2; return Math.max(0,m); };
  const Sint = ns => { if(!ns.length)return.5; let b=0; ns.forEach(n=>b+=popcount(n.s)); const p=b/(ns.length*32); return(p<=.001||p>=.999)?.001:-p*Math.log2(p)-(1-p)*Math.log2(1-p); };
  const Nvia = ns => ns.filter(n=>n.e<0&&memF(n.s)>.2).length;
  const calcPhi = h => { if(h.length<3)return 0; const r=h.slice(-15),m=r.reduce((a,b)=>a+b,0)/r.length,v=r.reduce((a,b)=>a+(b-m)**2,0)/r.length; return Math.tanh(.4/(Math.sqrt(v)+.05)); };
  const calcRisk = (g,p,e) => clamp((g<.4?.3:0)+(p>.6?.3:0)+(e==='anxious'?.2:0)+(e==='meltdown'?.4:0),0,1);
  const attempt = useCallback(raw=>{
    const a=ag.current, sem=semRef.current, tgt=phrase;
    const sim=stringSim(raw,tgt), need=sim<.78, cor=need?tgt:raw;
    const att={ts:Date.now(),tgt,raw,cor,need,sim};
