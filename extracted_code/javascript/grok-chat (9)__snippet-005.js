     82    const heartMode = heart?.mode;
     83 +  const heartName = heart?.heart_name || heart?.heartName || "Neurocohe
         rence Lattice";
     84    const heartStress = heart?.stress;
        ⋮
    253            <div>
    253 -            <div className="text-xs text-slate-400 uppercase tracking-w
         ide mb-1">Crystalline Heart</div>
    254 +            <div className="text-xs text-slate-400 uppercase tracking-w
         ide mb-1">{heartName}</div>
    255              <div className="text-lg font-semibold text-brand-black lead
         ing-tight">{heartMode || 'Unknown'}</div>

