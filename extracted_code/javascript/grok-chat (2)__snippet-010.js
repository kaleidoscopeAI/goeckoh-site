  const simAtt = ()=>{ let r=phrase; if(Math.random()<.35){const i=Math.floor(Math.random()*r.length);r=r.slice(0,i)+(r[i]?.match(/[aeiou]/i)?'aeiou'[Math.floor(Math.random()*5)]:'')+r.slice(i+1);} setInput(r); attempt(r); };
  const step = useCallback(()=>{
    const a=ag.current, sem=semRef.current;
    a.t++; const T=Math.max(.05,1-a.t*.003);
