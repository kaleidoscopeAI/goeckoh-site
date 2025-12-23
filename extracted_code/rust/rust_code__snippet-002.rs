fn bio_audio(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BioAcousticEngine>()?;
    Ok(())
