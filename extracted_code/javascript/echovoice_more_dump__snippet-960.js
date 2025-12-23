// ... Previous viz

// Pruning risk: alpha fade
const alpha = p5.map(particle.pruneRisk, 0, 1, 255, 50);
p5.fill(hue, 100, 100, alpha);

// Dashboard numeric overlays (for top 3 A nodes)
if (particle.a > 0.1) { // Select high A
  p5.fill(255);
  p5.textSize(10);
  p5.text(`ID:${particle.id.slice(0,4)} E:${particle.energy.toFixed(1)} K:${particle.k.toFixed(1)}`, projX + size, projY);
  p5.text(`A:${particle.a.toFixed(1)} Rep:${particle.repProb.toFixed(1)}`, projX + size, projY + 12);
}
