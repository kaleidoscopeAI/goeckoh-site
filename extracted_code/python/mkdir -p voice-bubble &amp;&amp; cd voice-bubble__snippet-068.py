In main.js, add this import at the top (right after the Three import):
import initWasm, { lattice_from_features } from "./wasm/pkg/goeckoh_lattice.js";

let wasmReady = false;
