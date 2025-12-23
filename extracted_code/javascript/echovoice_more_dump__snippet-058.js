if let Ok(mut r) = CPU_REGISTRY.lock() {
if let Some(core) = r.get_mut(&core_id) {
