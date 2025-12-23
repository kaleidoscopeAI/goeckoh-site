fn run_validation(config: ValidationConfig) -> Result<(), CrystalError> {
    println!("=== CHEMICAL CONSTRAINT VALIDATION ===");
    
    let graphs = load_or_process_dataset(&config.dataset_path, "")?;
    
    let sample_size = config.samples.min(graphs.len());
    let sample_graphs = &graphs[0..sample_size];
    
    let constraints = ChemicalConstraints::default();
    let mut errors: Vec<String> = Vec::new();
    for (i, g) in sample_graphs.iter().enumerate() {
        if let Err(errs) = constraints.validate(g) {
            for e in errs {
                errors.push(format!("graph {}: {}", i, e));
            }
        }
    }
    println!("\nValidation Results:");
    if errors.is_empty() {
        println!("All sampled graphs passed validation.");
    } else {
        for e in &errors { println!("- {}", e); }
    }
    
    if let Some(output_path) = &config.output_path {
        let json = serde_json::to_string_pretty(&errors)?;
        std::fs::write(output_path, json)?;
        info!("Validation report exported to {}", output_path);
    }
    
    Ok(())
