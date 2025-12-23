    250 +                const norm = Math.max(-1, Math.min(1, v));
    251 +                const hue = norm > 0 ? '34,197,94' : '239,68,68'; // gr
         een vs red
    252 +                const alpha = Math.min(0.9, Math.abs(norm));
    253 +                return (
    254 +                  <div
    255 +                    key={i}
    256 +                    style={{ flex: 1, background: `linear-gradient(180d
         eg, rgba(${hue},${alpha}) 0%, rgba(241,245,249,0) 90%)` }}
    257 +                  />
    258 +                );
    259 +              }) : <div className="text-sm text-slate-400 px-3 py-4">No
          lattice data yet.</div>}
    260 +            </div>
    261 +          </div>
    262 +        </div>
    263 +      </Card>
    264 +
    265        {/* Fragments selection */}

