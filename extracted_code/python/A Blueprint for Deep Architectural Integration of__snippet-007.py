initialize S = initial_cognitive_state
initialize c = initial_curiosity_tension = 0.0

for k in range(max_steps):
    uncertainty = compute_uncertainty(S)
    c = max(0, c + alpha * uncertainty - beta * signaling_feedback(k))  # curiosity tension update
    
    if c > threshold:
        crawl_output = O_crawl.process(c).await  # async crawl triggered
        S = G(S, crawl_output)  # integrate new knowledge
    
    S = G(S)  # normal cognitive update
    
    log_state(S, c)
