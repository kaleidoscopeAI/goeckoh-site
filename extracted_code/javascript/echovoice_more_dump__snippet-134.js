fn print_system_info() {
    let mut sys = System::new_all();
    sys.refresh_all();
    
    println!("=== SYSTEM INFORMATION ===");
    println!("OS: {}", System::long_os_version().unwrap_or_else(|| "Unknown".to_string()));
    println!("Total Memory: {} GB", sys.total_memory() / 1024 / 1024 / 1024);
    println!("Available Memory: {} GB", sys.available_memory() / 1024 / 1024 / 1024);
    println!("CPU Cores: {}", sys.cpus().len());
    if let Some(cpu) = sys.cpus().first() {
        println!("CPU Model: {}", cpu.brand());
        println!("CPU Frequency: {:.1} MHz", cpu.frequency());
    }
    println!("Rayon Threads: {}", rayon::current_num_threads());
    println!();
