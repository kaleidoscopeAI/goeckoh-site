print(f"--- Cycle {t+1} ---")
print(f"System Energy: {ham.energy(state):.4f}")

# ... (summarize state and get ai_thought from Ollama) ...
ai_thought = cognitive_engine.reflect_with_ollama(prompt)
visualizer.update_from_thought(ai_thought)

# 🧠 NEW: Apply cognitive feedback
# The total dimension of X_bar is node_count * vec_dim
total_vec_dim = node_count * vec_dim 
nudge_vector = thought_to_vector(ai_thought, total_vec_dim)

# Apply the nudge to the Hamiltonian's target state
ham.X_bar += nudge_vector 
# Normalize to prevent drift
ham.X_bar /= (np.linalg.norm(ham.X_bar) + 1e-9) 

print("[Cognitive Engine] Feedback applied. Hamiltonian target state has been nudged.")
time.sleep(1)

