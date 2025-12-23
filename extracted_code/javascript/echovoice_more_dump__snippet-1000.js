fn build_graph_from_image(&self, path: &Path) -> Result<MutagEntry, ProcessingError> {
    let img = image::open(path).map_err(|e| ProcessingError::IoError(e.into()))?;
    let (width, height) = img.dimensions();
    let mut x = Vec::with_capacity((width * height) as usize);
    let mut edge_index = Vec::new();
    let mut edge_attr = Vec::new();

    for y in 0..height {
        for x_pos in 0..width {
            let pixel = img.get_pixel(x_pos, y);
            let features = vec![
                pixel[0] as f64 / 255.0,
                pixel[1] as f64 / 255.0,
                pixel[2] as f64 / 255.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ];
            x.push(features);

            let idx = (y * width + x_pos) as usize;
            // Connect to right and below neighbors
            if x_pos + 1 < width {
                let right_idx = (y * width + x_pos + 1) as usize;
                edge_index.push(vec![idx, right_idx]);
                edge_index.push(vec![right_idx, idx]);
                edge_attr.push(vec![1.0, 0.0, 0.0, 0.0]);
                edge_attr.push(vec![1.0, 0.0, 0.0, 0.0]);
            }
            if y + 1 < height {
                let below_idx = ((y + 1) * width + x_pos) as usize;
                edge_index.push(vec![idx, below_idx]);
                edge_index.push(vec![below_idx, idx]);
                edge_attr.push(vec![1.0, 0.0, 0.0, 0.0]);
                edge_attr.push(vec![1.0, 0.0, 0.0, 0.0]);
            }
        }
    }

    Ok(MutagEntry { x, edge_index, edge_attr, y: 0 })
}
