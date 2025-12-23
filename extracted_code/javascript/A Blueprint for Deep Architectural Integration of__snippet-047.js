pub async fn process(&self, input: &[f32]) -> Vec<f32> {
    let (topic, cred_bias, depth) = self.decode_crawl_input(input);
    let queries = self.generate_queries(&topic).await;
    let data_batch = self.execute_crawl(queries, cred_bias, depth).await;
    let processed_batch = self.process_batch(data_batch).await;
    self.embed_to_output(processed_batch)
}

async fn generate_queries(&self, topic: &str) -> Vec<String> {
    // Call to LLM to get search queries ...
}

async fn execute_crawl(&self, queries: Vec<String>, bias: f32, depth: usize) -> Vec<RawData> {
    // Async crawl with concurrency, politeness, filtering
    let mut tasks = vec![];
    for query in queries {
        let task = task::spawn(async move {
            let url_list = self.url_frontier.collect_for_query(&query, bias, depth);
            for url in url_list {
                let resp = surf::get(&url).await?;
                let body = resp.body_string().await?;
                // Extract links or text ...
            }
            Ok::<_, surf::Error>(())
        });
        tasks.push(task);
    }
    // Await all
    futures::future::join_all(tasks).await;
    // Aggregate and return data
    vec![]
}

async fn process_batch(&self, data: Vec<RawData>) -> Vec<ProcessedData> {
    // Clean, summarize raw HTML, call summarizer LLM ...
}

fn embed_to_output(&self, data: Vec<ProcessedData>) -> Vec<f32> {
    // Embed processed info into vector output
}

fn decode_crawl_input(&self, input: &[f32]) -> (String, f32, usize) {
    // Interpret curiosity topic, bias, max crawling depth
}
