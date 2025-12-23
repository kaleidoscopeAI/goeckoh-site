// ... Line with thickness bond.w * 5, color map stress (from energy?)
const stress = (bond.k - 1) / 7; // Approx
const chue = p5.map(stress, 0, 1, 240, 0); // Blue to red
p5.stroke(chue, 100, 100);
p5.strokeWeight(bond.w * 5);
// Line...
// Particle flow: 3 particles along line
for (let t = 0; t < 1; t += 0.33) {
  const px = p5.lerp(projX1, projX2, t + Math.sin(p5.frameCount * 0.1) * 0.1);
  const py = p5.lerp(projY1, projY2, t + Math.sin(p5.frameCount * 0.1) * 0.1);
  p5.fill(255, 50);
  p5.ellipse(px, py, 1, 1);
}
