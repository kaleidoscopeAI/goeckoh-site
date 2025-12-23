      const ns=prev.map(n=>({...n,x:[...n.x]}));
      for(let u=0;u<ns.length;u++){
        let g=[0,0,0]; const nu=ns[u];
        for(const[ea,eb]of edges){if(ea!==u&&eb!==u)continue;const v=ea===u?eb:ea,nv=ns[v],d=manhattanDist(nu.x,nv.x)+.01,s=hammingSim(nu.s,nv.s);for(let i=0;i<3;i++){const df=nu.x[i]-nv.x[i],dr=df>0?1:df<0?-1:0;g[i]+=K*(d-L0)*dr/d+GAMMA*s*d*Math.exp(-d*d/2)*dr/d;}};for(let i=0;i<3;i++)g[i]+=2*LAMBDA*nu.x[i];
        const eta=.01,noise=Math.sqrt(2*eta*T);for(let i=0;i<3;i++){nu.x[i]-=eta*g[i]+noise*(Math.random()-.5)*.4;nu.x[i]=clamp(nu.x[i],-10,10);}
        const bi=Math.floor(Math.random()*32),oS=nu.s,nS=(oS^(1<<bi))>>>0;let dE=0;for(const[ea,eb]of edges){if(ea!==u&&eb!==u)continue;const v=ea===u?eb:ea;dE+=-J*(hammingSim(nS,ns[v].s)-hammingSim(oS,ns[v].s));}if(Math.random()<1/(1+Math.exp(dE/T)))nu.s=nS;
