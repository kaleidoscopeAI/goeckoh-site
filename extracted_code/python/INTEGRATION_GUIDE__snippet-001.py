voiceData ? {
  energy: voiceData.metrics.energy,
  f0: voiceData.metrics.f0,
  zcr: 0.3, // Calculate from audio
  tilt: 0.5,
  hnr: 0.8
} : null
