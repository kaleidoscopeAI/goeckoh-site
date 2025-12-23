const r=Math.min(1.0, Math.sqrt(x*x+y*y+z*z)); const t=r; const R=0.2+0.8*t; const G=0.5+0.5*(1.0-Math.abs(t-0.5)*2.0); const B=1.0-0.8*t;
colors[i]=R; colors[i+1]=G; colors[i+2]=B;} geom.attributes.color.needsUpdate=true;}
