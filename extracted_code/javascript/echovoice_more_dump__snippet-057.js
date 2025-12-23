pub extern \"system\" fn jni_verify_security(env: &JNIEnv, a: JFloatArray, b: JFloatArray, threshold: f32) -> bool {
let la = match env.get_array_length(a) { Ok(l) => l, Err(_) => 0 };
let lb = match env.get_array_length(b) { Ok(l) => l, Err(_) => 0 };
let mut va = vec![0f32; la as usize];
let mut vb = vec![0f32; lb as usize];
