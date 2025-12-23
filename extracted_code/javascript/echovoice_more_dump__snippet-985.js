let file = std::fs::File::open(path).map_err(CrystalError::from)?;
let df = ParquetReader::new(file).finish().map_err(CrystalError::from)?;
println!("=== DATASET INSPECTION ===");
println!("Path: {}", path);
println!("Rows: {}  Cols: {}", df.height(), df.width());
println!("Columns: {:?}", df.get_column_names());
println!("DTypes:   {:?}", df.dtypes());

// Show head
println!("\nFirst row preview:\n{:?}", df.head(Some(1)));

for name in ["x", "edge_index", "edge_attr", "y"] {
    if let Ok(col) = df.column(name) {
        println!("\nColumn '{}' dtype: {:?}", name, col.dtype());
        println!("  head: {:?}", col.head(Some(1)));
    }
}

Ok(())
