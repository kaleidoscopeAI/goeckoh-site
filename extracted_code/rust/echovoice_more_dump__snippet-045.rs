fn from(e: ProcessingError) -> Self {
    match e {
        ProcessingError::InvalidFormat(s) => CrystalError::Parameter(format!("Invalid format: {}", s)),
        ProcessingError::UnsupportedFormat(s) => CrystalError::Parameter(format!("Unsupported format: {}", s)),
        ProcessingError::DataError(s) => CrystalError::Data(s),
        ProcessingError::IoError(e) => CrystalError::IO(e),
    }
}
