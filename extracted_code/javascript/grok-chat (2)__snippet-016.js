  useEffect(()=>{let i;if(running)i=setInterval(step,150);return()=>clearInterval(i);},[running,step]);
  useEffect(()=>{const i=setInterval(()=>setAngle(a=>a+.01),50);return()=>clearInterval(i);},[]);
