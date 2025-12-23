294                Top lattice peaks: {Array.isArray((heart as any)?.heart_topk) ? (heart as any).heart_topk.length : 0}
295 +              <div className="flex gap-1 mt-1">
296 +                {(heart as any)?.heart_topk?.slice?.(0, 10)?.map?.((p: any, idx: number) => {
297 +                  const val = typeof p?.value === 'number' ? p.value : 0;
298 +                  const h = Math.min(32, Math.max(4, Math.abs(val) * 12));
299 +                  const color = val >= 0 ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)';
300 +                  return <div key={idx} style={{ width: 6, height: h, background: color }} title={`#${p?.index ?? idx}:
      ${val.toFixed?.(3) ?? val}`} />;
301 +                })}
302 +              </div>
303              </div>

