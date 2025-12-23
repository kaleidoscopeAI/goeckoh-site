const inter = p5.map(y, 0, gb.height, 0, 1) * (0.5 + 0.5 * avgB); // b-driven
const c = p5.lerpColor(fromColor, toColor, inter + Math.sin(p5.frameCount * 0.01 + y * 0.01) * 0.1);
gb.fill(c);
gb.rect(0, y, gb.width, 1);
