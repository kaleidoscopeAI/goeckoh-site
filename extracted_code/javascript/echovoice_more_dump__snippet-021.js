let len = env.get_array_length(input).unwrap_or(0);
let mut buf: Vec<f32> = vec![0.0; len as usize];
