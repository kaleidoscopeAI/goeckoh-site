const data = JSON.parse(event.data);
// Update voice data
setVoiceData(processVoiceData(data));
