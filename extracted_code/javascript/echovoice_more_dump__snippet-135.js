fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp_millis()
        .init();

    let cli = Cli::parse();

    if cli.system_info {
        print_system_info();
    }

    if let Err(e) = run(cli) {
        error!("Error: {}", e);
        std::process::exit(1);
    }
