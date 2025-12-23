    const c=canvasRef.current;if(!c||!nodes.length)return;const ctx=c.getContext('2d');ctx.fillStyle='#020208';ctx.fillRect(0,0,240,180);
    const proj=(x,y,z)=>{const cs=Math.cos(angle),sn=Math.sin(angle),rx=x*cs-z*sn,rz=x*sn+z*cs,sc=100/(rz+9);return{px:120+rx*sc,py:90+y*sc,d:rz};};
    ctx.strokeStyle='rgba(40,60,120,.08)';for(const[u,v]of edges){const pu=proj(nodes[u].x[0],nodes[u].x[1],nodes[u].x[2]),pv=proj(nodes[v].x[0],nodes[v].x[1],nodes[v].x[2]);ctx.beginPath();ctx.moveTo(pu.px,pu.py);ctx.lineTo(pv.px,pv.py);ctx.stroke();}
    const ps=nodes.map(n=>({...proj(n.x[0],n.x[1],n.x[2]),s:n.s,e:n.e,m:memF(n.s)})).sort((a,b)=>a.d-b.d);
    for(const p of ps){const sz=Math.max(1.5,3-p.d*.12),h=ds.risk>.6?0:ds.gcl>.6?120:(p.s%360);ctx.fillStyle=hsla(${h},${p.e<0?60:25}%,${35+p.m*18}%,${.5+p.m*.4});ctx.beginPath();ctx.arc(p.px,p.py,sz,0,Math.PI*2);ctx.fill();}
