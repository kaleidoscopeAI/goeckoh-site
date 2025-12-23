// ===== GOECKOH LATTICE WASM =====
let wasmReady = false;
let lattice_from_features = null;

(async () => {
  try {
    const m = await import("./wasm/pkg/goeckoh_lattice.js");
    await m.default();  // initWasm()
    lattice_from_features = m.lattice_from_features;
    wasmReady = true;
    console.log("✅ Lattice core loaded");
  } catch(e) { console.warn("WASM:", e); }
})();
