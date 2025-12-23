const connection = connectVoicePipeline({
  wsUrl: 'ws://localhost:8765',
  onVoiceData: (data) => {
    // Process voice data
    setVoiceMetrics(data);
  },
  onError: (error) => {
    console.error('Voice pipeline error', error);
  }
});

return () => connection.disconnect();
