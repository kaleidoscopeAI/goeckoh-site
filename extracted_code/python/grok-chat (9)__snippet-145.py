 90        final wavData = AudioUtils.createWavFile(_audioChunks, sampleRate);
 91 -
 92 -      // Send audio to backend for processing and get response
 93 -      final backendResponse = await _apiService.processAudio(wavData);
 94 -      final text = backendResponse['text'] ?? ""; // Backend should return processed text
 95 -      final audioPlaybackPath = backendResponse['audio_playback_path']; // Optional: backend returns audio file path
 96 -
 97 -      if (text.trim().length > 2) {
 98 -        // If backend returns audio, play it directly. Otherwise, use local TTS.
 99 -        if (audioPlaybackPath != null) {
100 -            // Implement audio playback from URL or local path here
101 -            // For now, just logging the path
102 -            print('Backend suggested audio playback from: $audioPlaybackPath');
103 -            // You would use a package like audioplayers or just play from URL directly
104 -        } else {
105 -            // Fallback to local TTS if no audio path from backend
106 -            _stateController.add(EchoState.SPEAKING);
107 -            await _tts.speak(text); // Use the text from backend for TTS
108 -            await _tts.awaitSpeakCompletion(true);
109 -        }
 91 +      final asr = await _apiService.asrRecognize(wavData);
 92 +      final text = (asr['text'] ?? "").toString();
 93 +      final backendResponse = await _apiService.processText(text);
 94 +      final reply = backendResponse['result'] is Map ? backendResponse['result']['response_text'] ?? text : text;
 95 +
 96 +      if (reply.trim().length > 0) {
 97 +        _stateController.add(EchoState.SPEAKING);
 98 +        await _tts.speak(reply);
 99 +        await _tts.awaitSpeakCompletion(true);
100        }

