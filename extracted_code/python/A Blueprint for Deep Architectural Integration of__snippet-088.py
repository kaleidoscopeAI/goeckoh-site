- **Activation**: When $b_k = 1$, $O_{\text{crawl}}$ is invoked via the routing matrix $R$ or internal logic.
- **Function**: It fetches web data based on queries generated from the curiosity state, processes content (e.g., summarization via $O_{\text{LLM}}$), and returns embeddings for integration.
- **Performance**: The crawl success reduces uncertainty, influencing $\mathrm{Perf}_k$ and thus future $b_{k+1}$.


