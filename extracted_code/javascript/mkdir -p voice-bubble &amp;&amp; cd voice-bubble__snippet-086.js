initWasm().then(() => {
  wasmReady = true;
  console.log("✅ Goeckoh Lattice loaded");
}).catch(e => console.warn("WASM init:", e));
