const [x, y, z] = particle.pos;
const projX = p5.map(x, -80, 80, 0, p5.width);
const projY = p5.map(y, -80, 80, 0, p5.height);
const size = p5.map(particle.a, 0, 1, 2, 20); // Size = awareness

// Color gradient: energy green-red
const hue = p5.map(particle.energy, 0, 3, 120, 0); // Green to red
p5.colorMode(p5.HSB);
p5.fill(hue, 100, 100, 200);
p5.ellipse(projX, projY, size, size);

// Surface glow/halo = knowledge
if (particle.k > 0.5) {
  p5.noFill();
  p5.stroke(hue, 50, 100, particle.k * 100);
  p5.ellipse(projX, projY, size * 1.5, size * 1.5);
}

// Fractal shimmer/ripple = mutation (flicker)
if (particle.mutation_sigma > 0.05) { // Assume added to snapshot
  p5.fill(255, 50);
  for (let i = 0; i < 3; i++) {
    p5.ellipse(projX + p5.random(-size/2, size/2), projY + p5.random(-size/2, size/2), 1, 1);
  }
}

// Directionally biased flow = perspective b
if (particle.b > 0) {
  p5.stroke(0, 0, 100, 50);
  const dir = p5.map(particle.b, -1, 1, -10, 10);
  p5.line(projX, projY, projX + dir, projY);
}

// Particle aura = semantic similarity (denser if aligned)
const sim = 0.8; // Approx from e_head
for (let i = 0; i < sim * 20; i++) {
  p5.fill(200, 50);
  p5.ellipse(projX + p5.random(-size, size), projY + p5.random(-size, size), 1, 1);
}

// Ring pulse = repProb
if (particle.repProb > 0.5) {
  p5.noFill();
  p5.stroke(100, 50, 100, particle.repProb * 100);
  const r = size + Math.sin(p5.frameCount * 0.1) * 5;
  p5.ellipse(projX, projY, r, r);
}

// Orbiting glyphs = concepts (tiny ellipses orbiting)
for (let i = 0; i < 3; i++) {
  const angle = p5.frameCount * 0.05 + i * 2 * Math.PI / 3;
  const gx = projX + Math.cos(angle) * size * 1.2;
  const gy = projY + Math.sin(angle) * size * 1.2;
  p5.fill(255, 100);
  p5.ellipse(gx, gy, 2, 2);
}
