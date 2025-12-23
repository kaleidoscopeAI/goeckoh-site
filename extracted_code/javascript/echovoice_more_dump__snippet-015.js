unsafe fn quantum_write(addr: *mut u32, qubits: &[Qubit]) {
let val = qubits.iter()
