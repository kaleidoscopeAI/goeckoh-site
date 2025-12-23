if !p.exists() && !url.is_empty() {
    ensure_dataset_exists(path, url)?;
}
// Now, if the file exists, load it, else error?
if !p.exists() {
    return Err(CrystalError::IO(std::io::Error::new(std::io::ErrorKind::NotFound, format!("File {} does not exist and no URL provided", path)));
}
load_mutag_dataset(path)
