fn goeckoh_audio_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AudioKernel>()?;
    Ok(())
