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

• I'm launching the server using the GOECKOH_LIGHT_MODE environment to reduce latency and test p95 under 200ms without
  fast_mode, aiming to balance performance with full computations.

