if let Ok(r) = CPU_REGISTRY.lock() {
if let Some(core) = r.get(&core_id) {
