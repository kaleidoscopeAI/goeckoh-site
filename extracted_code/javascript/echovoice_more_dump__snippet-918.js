<div className="w-full h-screen bg-black relative overflow-hidden">
  <div ref={mountRef} className="absolute inset-0" />

  <div className="absolute top-4 left-4 bg-black/60 p-3 rounded-lg text-cyan-300 text-sm space-y-1">
    <div>🧠 AI Thought: {aiThought.slice(0, 80)}...</div>
    <div>💬 Response: {aiResponse.slice(0, 120)}...</div>
    <div>🌀 Coherence: {metrics.coherence.toFixed(2)}</div>
    <div>❤️ Valence: {metrics.valence.toFixed(2)}</div>
  </div>

  <div className="absolute bottom-4 left-1/2 -translate-x-1/2 w-3/4 max-w-2xl bg-black/60 rounded-lg p-4">
    <input
      type="text"
      placeholder="Ask the AI to describe, imagine, or tell a story..."
      className="w-full bg-transparent text-white border border-cyan-400 rounded p-2"
      onKeyDown={async (e) => {
        if (e.key === 'Enter') {
          const query = e.target.value.trim();
          if (query.length > 0) {
            const thought = await fetchAIResponse(query);
            setMetrics((m) => ({
              ...m,
              curiosity: Math.min(0.9, m.curiosity + 0.05),
              coherence: Math.min(0.9, m.coherence + 0.03),
              valence: (m.valence + Math.random() * 0.1) % 1.0,
            }));
            e.target.value = '';
          }
        }
      }}
    />
  </div>
</div>
