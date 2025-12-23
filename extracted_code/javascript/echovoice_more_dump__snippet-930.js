const clipped = Math.max(-0.5, Math.min(0.5, delta[d]));
node.sem[d] += engine.llm_eta * clipped;
