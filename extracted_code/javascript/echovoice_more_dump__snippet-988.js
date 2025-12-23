let config = if let Some(config_path) = &cli.config {
    CrystalConfig::from_file(config_path)?
} else {
    CrystalConfig::default()
};

match cli.command {
    Commands::Run { 
        dataset, url, test_ratio, k_folds, seed, output, 
        multi_objective, adaptive_cooling, chemical_validation,
        evolution_generations, population_size 
    } => {
        run_experiment(RunConfig {
            dataset_path: dataset,
            dataset_url: url,
            test_ratio,
            k_folds,
            seed,
            output_path: output,
            multi_objective,
            adaptive_cooling,
            chemical_validation,
            evolution_generations,
            population_size,
            config,
        })
    },
    Commands::Evolve { dataset, generations, population_size, output, seed } => {
        let evo_cfg = EvolutionConfig {
            dataset_path: dataset,
            generations,
            population_size,
            seed,
            ..Default::default()
        };
        run_evolution(evo_cfg, output)
    },
    Commands::Validate { dataset, samples, output } => {
        run_validation(ValidationConfig {
            dataset_path: dataset,
            samples,
            output_path: output,
            config,
        })
    },
    Commands::Benchmark { dataset, iterations, output } => {
        run_benchmark(BenchmarkConfig {
            dataset_path: dataset,
            iterations,
            output_path: output,
            config,
        })
    },
    Commands::Config { output } => {
        generate_config(&output)
    },
    Commands::InspectDataset { dataset, url } => {
        ensure_dataset_exists(&dataset, &url)?;
        inspect_parquet(&dataset)
    },
}
