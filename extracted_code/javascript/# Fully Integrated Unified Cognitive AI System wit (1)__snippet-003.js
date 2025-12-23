if (!nodes.length) return;

// Example: Map 1 node per pixel, assuming width * height == nodes.length
// Aggregate vector features into colors (normalized)
const data = new Uint8ClampedArray(width * height * 4);

// Example PCA-like feature extraction (simplified)
nodes.forEach((node, i) => {
  const baseIdx = i * 4;
  // Use first 3 dims of vector as RGB after normalizing [0,1]
  for (let j = 0; j < 3; j++) {
    data[baseIdx + j] = Math.min(255, Math.max(0, node.vector[j] * 255));
  }
  // Alpha channel based on arousal or valence
  data[baseIdx + 3] = Math.min(255, Math.max(50, node.arousal * 255));
});

setImageData(data);
