1063 |     let results: Vec<_> = graphs.par_iter_mut().enumerate().map(|(i, graph)| -> Result<_> {
     |         ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_results`

