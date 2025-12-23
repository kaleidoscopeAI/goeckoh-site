      for(const n of ns){let le=0;for(const[ea,eb]of edges){if(ea===n.id||eb===n.id){const v=ea===n.id?eb:ea;le+=-hammingSim(n.s,ns[v].s);}}n.e=le;}
      const si=Sint(ns),nv=Nvia(ns);
