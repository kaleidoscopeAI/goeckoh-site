S_k = initial_cognitive_state()
c_k = 0.0  # curiosity tension

for k in range(max_steps):
    uncertainty = compute_uncertainty(S_k)
    c_k = max(0, 0.9 * c_k + 0.1 * uncertainty - 0.05 * performance_feedback())
    
    if c_k > threshold:
        I_crawl = crawl_module.process(c_k)  # async trusted call
    else:
        I_crawl = zero_vector()
    
    S_k = cognitive_update(S_k, I_crawl)
    log_state(S_k, c_k)
