    let result = self.executor.execute(&action).await?;

    // 3. Record result & update graph embeddings
    { 
        let mut tasks = self.executed_tasks.lock().await;
        tasks.push(result.clone());
    }
    let mut graph = self.graph.lock().await;
    graph.update_embeddings(&result); 

