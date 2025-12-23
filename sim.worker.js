// src/workers/sim.worker.js

// This is where the heavy simulation logic would go,
// offloading it from the main UI thread.

self.onmessage = (e) => {
  // Receive state from main thread
  const { nodes } = e.data;

  // ... perform heavy computation ...
  const updatedNodes = nodes.map(n => ({...n, x: n.x + 0.1}));

  // Send back the updated state
  self.postMessage({ nodes: updatedNodes });
};
