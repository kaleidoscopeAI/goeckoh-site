const draw = (p5: any) => {
  // Aurora gradients: offscreen for smooth
  if (!this.gradientBuffer) {
    this.gradientBuffer = p5.createGraphics(p5.width, p5.height);
  }
  const gb = this.gradientBuffer;
  gb.clear();
  gb.noStroke();
  const avgB = frame.particles.reduce((sum, p) => sum + p.b, 0) / frame.particles.length;
  const fromColor = p5.color(0, 50, 100, 50); // Blue-ish
  const toColor = p5.color(120, 50, 100, 50); // Green-ish
  for (let y = 0; y < gb.height; y++) {
    const inter = p5.map(y, 0, gb.height, 0, 1) * (0.5 + 0.5 * avgB); // b-driven
    const c = p5.lerpColor(fromColor, toColor, inter + Math.sin(p5.frameCount * 0.01 + y * 0.01) * 0.1);
    gb.fill(c);
    gb.rect(0, y, gb.width, 1);
  }
  p5.image(gb, 0, 0);

  frame.particles.forEach((particle) => {
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
  });

  // ... Bonds, etc.
