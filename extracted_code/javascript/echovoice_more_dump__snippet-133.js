fn ensure_dataset_exists(path: &str, url: &str) -> Result<(), CrystalError> {
    let p = Path::new(path);
    if p.exists() {
        return Ok(());
    }
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    let mut resp = reqwest::blocking::get(url)?;
    resp.copy_to(&mut file)?;
    Ok(())
