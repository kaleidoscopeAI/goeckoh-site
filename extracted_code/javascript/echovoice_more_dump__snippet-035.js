let qs: Vec<Qubit> = buf.into_iter().map(|a| Qubit{amp: a}).collect();
let reg_opt = CPU_REGISTRY.lock().unwrap().get(&(core_id as usize)).cloned();
if let Some(mut core) = reg_opt {
let qreg = QuantumRegister { qubits: qs.clone(), hw_map: core.msr_mapping.clone() };
let _ = core.apply_quantum_state(&qreg);
