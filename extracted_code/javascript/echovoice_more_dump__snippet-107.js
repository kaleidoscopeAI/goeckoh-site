const avgV = frame.particles.reduce((sum, p) => sum + p.v, 0) / frame.particles.length;
const fromC = p5.color(avgV > 0 ? 120 : 0, 100, 100, 50); // Green/red valence
