- $S_{k+1}$ is the updated cognitive state.
- $G$ is the global integration function that combines outputs from all thought engines.
- $O_i$ are the thought engines, including:
    - $O_{\text{LLM}}$: A Hugging Face transformer for reasoning (e.g., LLaMA or Mistral), implemented via GGUF and llama.cpp.
    - $O_{\text{embedding}}$: A Hugging Face embedding model (e.g., BGE), used to convert text to vectors.
    - $O_{\text{crawl}}$: The web crawler engine, activated when $b_k = 1$.
    - Other engines for perception, action, etc.
- $P_i$ are projection operators that select relevant parts of $X_k$ for each engine.
- $R$ is the routing matrix that determines connectivity between engines.
- $I_k$ is external input.
- $\eta_k$ is noise.
- $b_{k+1}$ is the updated curiosity bit, derived from the threshold function.
- $\rho, \lambda, \sigma, \beta, \theta$ are parameters: $\rho$ is the decay factor, $\lambda$ scales the bit to tension, $\sigma$ weights performance feedback, $\beta$ weights external drive $u_k$, and $\theta$ is the threshold.
- $\mathrm{Perf}_k$ is performance feedback (e.g., reduction in cognitive uncertainty).
- $u_k$ is an external curiosity drive signal.


