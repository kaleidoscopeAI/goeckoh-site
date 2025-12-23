fn export_results(summary: &RunSummary, path: &str) -> Result<(), CrystalError> {
    let json = serde_json::to_string_pretty(summary)?;
    std::fs::write(path, json)?;
    Ok(())
