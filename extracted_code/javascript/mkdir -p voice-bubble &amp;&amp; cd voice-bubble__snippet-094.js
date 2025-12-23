fn from_slice(slice: &[f32]) -> Self {
    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<f32>(),
        )
    }.to_vec();

    Self {
        data: Bytes::from(bytes),
        timestamp: std::time::Instant::now().elapsed().as_micros() as u64,
    }
}
