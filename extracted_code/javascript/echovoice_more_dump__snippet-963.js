const angle = p5.frameCount * 0.05 + i * 2 * Math.PI / p.symbols.length;
p5.text(sym, projX + Math.cos(angle) * size * 1.2, projY + Math.sin(angle) * size * 1.2);
