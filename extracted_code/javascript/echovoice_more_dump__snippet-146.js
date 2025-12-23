let processed = runtime.block_on(engine.process(p)).map_err(|e| {
    match e {
        ProcessingError::InvalidFormat(s) => CrystalError::Parameter(format!("Invalid format: {}", s)),
        // ... etc.
    }
