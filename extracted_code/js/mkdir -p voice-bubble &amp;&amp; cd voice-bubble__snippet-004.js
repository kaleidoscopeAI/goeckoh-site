let wasmReady = false;
try {
  import("./wasm/pkg/goeckoh_lattice.js").then(module => {
    initWasm().then(() => {
      wasmReady = true;
      console.log("✅ Goeckoh Lattice loaded");
    }).catch(e => console.warn("WASM init:", e));
  });
} catch(e) {
  console.warn("No WASM support:", e);
}
