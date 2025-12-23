fn goeckoh_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AudioEngine>()?;
    Ok(())
