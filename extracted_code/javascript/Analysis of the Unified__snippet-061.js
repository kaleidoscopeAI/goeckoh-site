// 1. Log the action and check resource limits
info!("Proposing action: {:?}", action); 

// 2. Execute the action and await the result
let result = self.executor.execute(&action).await?;

// 3. Record result & update graph embeddings
{ 
    let mut tasks = self.executed_tasks.lock().await;
    tasks.push(result.clone());
}
let mut graph = self.graph.lock().await;
graph.update_embeddings(&result); 

