fn generate_config(output_path: &str) -> Result<(), CrystalError> {
    let config = CrystalConfig::default();
    save_config(&config, output_path)?;
    println!("Default configuration written to {}", output_path);
    
    Ok(())
