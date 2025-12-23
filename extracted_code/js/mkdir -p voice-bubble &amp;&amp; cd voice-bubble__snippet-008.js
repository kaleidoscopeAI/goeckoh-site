// After workletNode creation:
monitorGainNode = audioCtx.createGain();
monitorGainNode.gain.value = 0.0;  // SAFE default
workletNode.connect(monitorGainNode);
monitorGainNode.connect(audioCtx.destination);
applyMonitor();  // Wire UI

// In stopAudio():
try { if (monitorGainNode) monitorGainNode.disconnect(); } catch(e){}
monitorGainNode = null;
