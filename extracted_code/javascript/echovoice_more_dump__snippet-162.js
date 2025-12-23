The add_node function returns a Result<usize>, so we must handle it with ? only if the function returns a Result. Change the function to return Result<Vec<Graph>>.

