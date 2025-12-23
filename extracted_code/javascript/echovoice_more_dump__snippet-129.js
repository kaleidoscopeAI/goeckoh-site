fn run_evolution(config: EvolutionConfig, output_path: Option<String>) -> Result<(), CrystalError> {
    println!("=== EMBEDDING PARAMETER EVOLUTION ===");
    
    let seed = config.seed.unwrap_or_else(generate_seed);
    let graphs = load_or_process_dataset(&config.dataset_path, "https://huggingface.co/datasets/graphs-datasets/MUTAG/resolve/main/train.parquet")?;

    let best_params = evolve_embedding_parameters(&config, &graphs)?;
    
    println!("\nBest embedding parameters found:");
    println!("{:#?}", best_params);
    
    if let Some(output_path) = &output_path {
        let json = serde_json::to_string_pretty(&best_params)?;
        std::fs::write(output_path, json)?;
        info!("Best parameters exported to {}", output_path);
    }
    
    Ok(())
