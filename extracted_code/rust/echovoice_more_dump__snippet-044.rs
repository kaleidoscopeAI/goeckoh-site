println!("{}  Accuracy:  {:>6.2}%", prefix, metrics.accuracy * 100.0);
println!("{}  Precision: {:>6.2}%", prefix, metrics.precision * 100.0);
println!("{}  Recall:    {:>6.2}%", prefix, metrics.recall * 100.0);
println!("{}  F1-Score:  {:>6.2}%", prefix, metrics.f1_score * 100.0);
