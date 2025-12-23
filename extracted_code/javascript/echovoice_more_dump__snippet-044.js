// echo transformed values using a small deterministic transform
let len = env.get_array_length(input).unwrap_or(0);
let mut buf = vec![0f32; len as usize];
