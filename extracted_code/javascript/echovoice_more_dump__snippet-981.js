pub fn new() -> Self { Self }

fn build_graph_from_image(&self, path: &Path) -> MutagEntry {
    // Use pygame to load image and create grid graph
    pygame::init();
    let img = pygame::image::load(path.to_str().unwrap()).expect("Failed to load image");
    let width = img.get_width() as usize;
    let height = img.get_height() as usize;
    let mut x = Vec::with_capacity(width * height);
    let mut edge_index = Vec::new();
    let mut edge_attr = Vec::new();

    for y in 0..height {
        for x_pos in 0..width {
            let idx = y * width + x_pos;
            let pixel = img.get_at((x_pos as i32, y as i32));
            let features = vec![
                pixel.r as f64 / 255.0,
                pixel.g as f64 / 255.0,
                pixel.b as f64 / 255.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ];
            x.push(features);

            // Connect to right and below
            if x_pos + 1 < width {
                edge_index.push(vec![idx, idx + 1]);
                edge_index.push(vec![idx + 1, idx]);
                edge_attr.push(vec![1.0, 0.0, 0.0, 0.0]);
                edge_attr.push(vec![1.0, 0.0, 0.0, 0.0]);
            }
            if y + 1 < height {
                edge_index.push(vec![idx, idx + width]);
                edge_index.push(vec![idx + width, idx]);
                edge_attr.push(vec![1.0, 0.0, 0.0, 0.0]);
                edge_attr.push(vec![1.0, 0.0, 0.0, 0.0]);
            }
        }
    }

    MutagEntry { x, edge_index, edge_attr, y: 0 }
}
