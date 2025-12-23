pub fn apply_qubits_to_cpu(env: &JNIEnv, core_id: i32, qubits_j: JFloatArray) -> jfloatArray {
let len = match env.get_array_length(qubits_j) {
