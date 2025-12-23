if !p.exists() {
    if !url.is_empty() {
        ensure_dataset_exists(path, url)?;
    } else {
        return Err(CrystalError::IO(std::io::Error::new(std::io::ErrorKind::NotFound, format!("File {} does not exist", path)));
    }
}
load_mutag_dataset(path)
