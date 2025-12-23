println!("\n=== RESULTS ===");

println!("Mode: {}", summary.setup.test_mode);
if summary.setup.k_folds > 0 {
    println!(" ({}-fold)", summary.setup.k_folds);
}

if let Some(features) = &summary.features_used {
    println!("Features: {}", features.join(", "));
}

println!("\nBaseline Performance:");
print_metrics("", &summary.baseline);

if let Some(enhanced) = &summary.enhanced {
    println!("\nEnhanced Performance:");
    print_metrics("", enhanced);

    if let Some(deltas) = &summary.deltas {
        println!("\nImprovements:");
        println!("  Accuracy:  {:>+6.2} pp", deltas.accuracy_pp);
        println!("  Precision: {:>+6.2} pp", deltas.precision_pp);
        println!("  Recall:    {:>+6.2} pp", deltas.recall_pp);
        println!("  F1-Score:  {:>+6.2} pp", deltas.f1_pp);
    }
}

if let Some(perf) = &summary.performance {
    println!("\nPerformance Metrics:");
    println!("  Total Time:      {:.2?}", perf.total_time);
    println!("  Annealing Time:  {:.2?}", perf.annealing_time);
    println!("  Memory Peak:     {:.2} MB", perf.peak_memory_mb);
    println!("  Energy Reduction: {:.4}", perf.avg_energy_reduction);
}
