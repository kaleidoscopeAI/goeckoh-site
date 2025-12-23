if (!nodes.length) return;
const data = new Uint8ClampedArray(width * height * 4);

nodes.forEach((node, i) => {
  const idx = i * 4;
  // Map first 3 vector components to RGB (normalized 0-255)
  for (let j = 0; j < 3; j++) {
    let val = node.vector[j];
    val = Math.min(1, Math.max(0, val)); // clamp between 0 and 1
    data[idx + j] = Math.floor(val * 255);
  }

  // Alpha from arousal (scaled)
  data[idx + 3] = Math.floor(Math.min(1, Math.max(0, node.arousal)) * 255);
});

setImageData(data);
