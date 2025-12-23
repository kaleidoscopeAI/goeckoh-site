fn audio_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed_py, m)?)?;
    Ok(())
