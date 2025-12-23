let p = Path::new(path);
if p.exists() {
    return Ok(());
}
if url.is_empty() {
    return Err(CrystalError::IO(std::io::Error::new(std::io::ErrorKind::NotFound, format!("File {} does not exist and no download URL provided", path)));
}
// ... download code
