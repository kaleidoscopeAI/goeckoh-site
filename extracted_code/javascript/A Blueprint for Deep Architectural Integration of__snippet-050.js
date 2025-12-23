pub async fn process(&self, input: &[f32]) -> Vec<f32> {
    let (topic, credibility_bias, depth) = self.decode_crawl_input(input);
    let queries = self.generate_queries(&topic).await;
    let data_batch = self.execute_crawl(queries, credibility_bias, depth).await;
    let processed_batch = self.process_batch(data_batch).await;
    self.embed_to_output(processed_batch)
}

// Additional methods as outlined in Section 3 provide querying,
// polite crawling, focused filtering, summarization, and embedding.
