 90    const heartEmo = heart?.emotional || {};
 91 +  const heartRust = heart?.heart_rust || {};
 92
    ⋮
288              </div>
289 +            <div className="text-xs text-slate-500 mt-1">
290 +              Rust lattice: Coherence {heartRust.coherence !== undefine
     d ? heartRust.coherence.toFixed?.(3) : '—'} · Arousal {heartRust.arousa
     l !== undefined ? heartRust.arousal.toFixed?.(3) : '—'} · Valence {hear
     tRust.valence !== undefined ? heartRust.valence.toFixed?.(3) : '—'}
291 +            </div>
292            </div>

