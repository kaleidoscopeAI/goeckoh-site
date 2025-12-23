  const suggestions = await llm.suggest(reflections);
  for (const id in suggestions) {
    const delta = suggestions[id];
    const predictedDE = Math.random() * 0.2 - 0.1; // Mock ΔE; real: finite diff
    if (predictedDE > 0) {
      // Flag for speculation
      const snap = engine.snapshot();
      const specResp = await fetch(process.env.SPEC_WORKER_URL + '/speculate', {
        method: 'POST',
        body: JSON.stringify({ snapshot: snap, target_node: id, suggested_delta: delta })
      });
      const { fold_delta, accepted } = await specResp.json();
      if (accepted) {
        applyDelta(id, fold_delta); // Function to apply clipped delta
      }
    } else {
      applyDelta(id, delta);
    }
  }
