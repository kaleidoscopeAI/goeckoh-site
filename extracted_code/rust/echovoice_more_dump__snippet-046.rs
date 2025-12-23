fn from(error: ProcessingError) -> Self {
    match error {
        ProcessingError::InvalidFormat(s) => CrystalError::Parameter(s),
        ProcessingError::UnsupportedFormat(s) => CrystalError::Parameter(s),
        ProcessingError::DataError(s) => CrystalError::Data(s),
        ProcessingError::IoError(e) => CrystalError::IO(e),
    }
}
