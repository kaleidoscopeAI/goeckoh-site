import React from 'react';

const LLMPanel = ({ llmState }) => {
  return (
    <div className="absolute bottom-4 left-4 bg-gray-800/50 p-4 rounded-lg text-white font-mono text-sm">
      <h2 className="font-bold text-lg mb-2">LLM State</h2>
      <p>Status: {llmState.status}</p>
      <p>Last Query: {llmState.lastQuery}</p>
      <p>Last Response: {llmState.lastResponse}</p>
    </div>
  );
};

export default LLMPanel;
