fn bio_audio(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BioAcousticEngine>()?;
    Ok(())
