fn run_experiment(config: RunConfig) -> Result<(), CrystalError> {
    let start_time = Instant::now();
    
    println!("=== COGNITIVE CRYSTAL v2.0 ===");
    println!("Advanced Physics-Inspired Graph Classification");
    println!();

    let seed = config.seed.unwrap_or_else(generate_seed);
    info!("Using seed: {}", seed);

    info!("Loading/Processing dataset from {}", config.dataset_path);
    let graphs = load_or_process_dataset(&config.dataset_path, &config.dataset_url)?;
    info!("Loaded {} graphs", graphs.len());

    let embedding_params = if config.evolution_generations > 0 {
        info!("Evolving embedding parameters for {} generations...", config.evolution_generations);
        let evolution_config = EvolutionConfig {
            dataset_path: config.dataset_path.clone(),
            generations: config.evolution_generations,
            population_size: config.population_size,
            seed: Some(seed ^ 0x1234),
            ..Default::default()
        };
        Some(evolve_embedding_parameters(&evolution_config, &graphs)?)
    } else {
        None
    };

    let mut schedule = config.config.annealing.clone();
    
    let adaptive_params = if config.adaptive_cooling {
        Some(AdaptiveCoolingParams::default())
    } else {
        None
    };

    let multi_objective_params = if config.multi_objective {
        Some(MultiObjectiveParams::default())
    } else {
        None
    };

    let summary = if config.k_folds >= 2 {
        evaluate_kfold_advanced(
            graphs,
            &mut schedule,
            config.k_folds,
            seed,
            adaptive_params,
            multi_objective_params,
            embedding_params,
            config.chemical_validation,
        )?
    } else {
        evaluate_holdout_advanced(
            graphs,
            &mut schedule,
            config.test_ratio,
            seed,
            adaptive_params,
            multi_objective_params,
            embedding_params,
            config.chemical_validation,
        )?
    };

    print_results(&summary);

    if let Some(output_path) = &config.output_path {
        export_results(&summary, output_path)?;
        info!("Results exported to {}", output_path);
    }

    let total_time = start_time.elapsed();
    println!("\nTotal execution time: {:.2?}", total_time);

    Ok(())
