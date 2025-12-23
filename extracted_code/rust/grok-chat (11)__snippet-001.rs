fn embed_py(py: Python, script: &str) -> PyResult<String> {
    py.run_bound(script, None, None)?;
    Ok("Executed".to_string())
