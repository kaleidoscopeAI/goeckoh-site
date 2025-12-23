pub fn register_cpu(core_id: i32, owner_mode: bool) {
let mut r = CPU_REGISTRY.lock().unwrap();
