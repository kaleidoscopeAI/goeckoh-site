// Use annealing / Hamming energy to pick optimal task
let mut rng = rand::thread_rng();
Action { description: format!("Task {}", rng.gen::<u32>()) }
