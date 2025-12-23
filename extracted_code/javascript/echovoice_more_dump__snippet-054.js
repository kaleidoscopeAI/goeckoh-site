pub extern \"system\" fn jni_compute_commitment(env: &JNIEnv, inputs: JFloatArray) -> jni::sys::jbyteArray {
let len = match env.get_array_length(inputs) {
