let json = serde_json::to_string_pretty(summary)?;
std::fs::write(path, json)?;
Ok(())
