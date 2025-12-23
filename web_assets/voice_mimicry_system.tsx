import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Volume2, Settings, AlertCircle } from 'lucide-react';

export default function VoiceMimicrySystem() {
  const [isListening, setIsListening] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [noiseReduction, setNoiseReduction] = useState(50);
  const [playbackDelay, setPlaybackDelay] = useState(500);
  const [lastSpoken, setLastSpoken] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('Ready');
  
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const processorRef = useRef(null);
  const isRecordingRef = useRef(false);

  useEffect(() => {
    return () => {
      cleanup();
    };
  }, []);

  const cleanup = () => {
    isRecordingRef.current = false;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
  };

  const startListening = async () => {
    setError('');
    setStatus('Requesting microphone access...');
    
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 44100
        } 
      });
      
      setStatus('Microphone access granted');
      streamRef.current = stream;
      
      // Create audio context
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      audioContextRef.current = new AudioContext();
      
      // Create analyser for visualization
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 2048;
      analyserRef.current.smoothingTimeConstant = 0.8;
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      
      // Create script processor for capturing audio
      const bufferSize = 4096;
      processorRef.current = audioContextRef.current.createScriptProcessor(bufferSize, 1, 1);
      
      const audioChunks = [];
      let isCapturing = false;
      let captureStartTime = 0;
      const captureDuration = 2000; // 2 seconds
      
      processorRef.current.onaudioprocess = (e) => {
        if (!isRecordingRef.current) return;
        
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Check for actual sound (not silence)
        const maxAmplitude = Math.max(...Array.from(inputData).map(Math.abs));
        
        if (maxAmplitude > 0.01) {
          if (!isCapturing) {
            isCapturing = true;
            captureStartTime = Date.now();
            audioChunks.length = 0;
            setStatus('Recording...');
          }
          
          // Store audio data
          audioChunks.push(new Float32Array(inputData));
        }
        
        // Process captured audio after duration
        if (isCapturing && (Date.now() - captureStartTime > captureDuration)) {
          isCapturing = false;
          setStatus('Processing...');
          
          if (audioChunks.length > 0) {
            playbackAudio(audioChunks);
          }
        }
      };
      
      // Connect audio graph
      source.connect(analyserRef.current);
      source.connect(processorRef.current);
      processorRef.current.connect(audioContextRef.current.destination);
      
      // Start visualization
      visualize();
      
      isRecordingRef.current = true;
      setIsListening(true);
      setStatus('Listening for speech...');
      
    } catch (err) {
      console.error('Microphone error:', err);
      setError(`Failed to access microphone: ${err.message}`);
      setStatus('Error');
      cleanup();
    }
  };

  const stopListening = () => {
    setStatus('Stopped');
    setIsListening(false);
    cleanup();
    setAudioLevel(0);
  };

  const visualize = () => {
    if (!analyserRef.current || !isRecordingRef.current) return;
    
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      if (!isRecordingRef.current) return;
      
      requestAnimationFrame(draw);
      analyserRef.current.getByteTimeDomainData(dataArray);
      
      // Calculate RMS for audio level
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        const normalized = (dataArray[i] - 128) / 128;
        sum += normalized * normalized;
      }
      const rms = Math.sqrt(sum / bufferLength);
      setAudioLevel(Math.min(100, rms * 200));
    };
    
    draw();
  };

  const playbackAudio = async (audioChunks) => {
    if (!audioContextRef.current) return;
    
    try {
      // Concatenate all chunks
      const totalLength = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const concatenated = new Float32Array(totalLength);
      let offset = 0;
      
      for (const chunk of audioChunks) {
        concatenated.set(chunk, offset);
        offset += chunk.length;
      }
      
      // Apply simple noise reduction
      const threshold = 0.01 * (1 - noiseReduction / 100);
      for (let i = 0; i < concatenated.length; i++) {
        if (Math.abs(concatenated[i]) < threshold) {
          concatenated[i] = 0;
        }
      }
      
      // Create audio buffer
      const audioBuffer = audioContextRef.current.createBuffer(
        1, 
        concatenated.length, 
        audioContextRef.current.sampleRate
      );
      audioBuffer.getChannelData(0).set(concatenated);
      
      // Play back after delay
      setTimeout(() => {
        if (!audioContextRef.current) return;
        
        const source = audioContextRef.current.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContextRef.current.destination);
        source.start(0);
        
        setStatus('Playing back...');
        
        source.onended = () => {
          if (isRecordingRef.current) {
            setStatus('Listening for speech...');
          }
        };
      }, playbackDelay);
      
    } catch (err) {
      console.error('Playback error:', err);
      setError(`Playback failed: ${err.message}`);
    }
  };

  const toggleListening = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl">
        {/* Main Control */}
        <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 shadow-2xl border border-white/20">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-white mb-2">Voice Mimicry System</h1>
            <p className="text-blue-200">Continuous voice capture and playback</p>
          </div>
          
          {/* Error Display */}
          {error && (
            <div className="bg-red-500/20 border border-red-400 rounded-lg p-4 mb-6 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-300 flex-shrink-0 mt-0.5" />
              <div className="text-red-100 text-sm">{error}</div>
            </div>
          )}
          
          {/* Microphone Button */}
          <div className="flex justify-center mb-6">
            <button
              onClick={toggleListening}
              className={`relative w-32 h-32 rounded-full transition-all duration-300 ${
                isListening 
                  ? 'bg-red-500 hover:bg-red-600 shadow-lg shadow-red-500/50' 
                  : 'bg-blue-500 hover:bg-blue-600 shadow-lg shadow-blue-500/50'
              }`}
            >
              {isListening ? (
                <MicOff className="w-16 h-16 text-white mx-auto" />
              ) : (
                <Mic className="w-16 h-16 text-white mx-auto" />
              )}
              
              {/* Pulse animation when listening */}
              {isListening && (
                <>
                  <div className="absolute inset-0 rounded-full bg-red-500 animate-ping opacity-25"></div>
                  <div className="absolute inset-0 rounded-full bg-red-400 animate-pulse opacity-20"></div>
                </>
              )}
            </button>
          </div>
          
          <div className="text-center text-white mb-8">
            <p className="text-xl font-semibold mb-2">
              {isListening ? 'Active' : 'Click to start'}
            </p>
            <p className="text-sm text-blue-200">{status}</p>
          </div>
          
          {/* Audio Level Visualization */}
          <div className="mb-8">
            <div className="flex items-center gap-2 mb-2">
              <Volume2 className="w-5 h-5 text-white" />
              <span className="text-white text-sm">Audio Level</span>
            </div>
            <div className="h-4 bg-white/20 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-green-400 via-yellow-400 to-red-500 transition-all duration-100"
                style={{ width: `${audioLevel}%` }}
              ></div>
            </div>
          </div>
          
          {/* Settings Toggle */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="w-full flex items-center justify-center gap-2 bg-white/10 hover:bg-white/20 text-white py-3 rounded-xl transition-colors"
          >
            <Settings className="w-5 h-5" />
            <span>{showSettings ? 'Hide' : 'Show'} Settings</span>
          </button>
        </div>
        
        {/* Settings Panel */}
        {showSettings && (
          <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 shadow-2xl border border-white/20 mt-4">
            <h2 className="text-2xl font-bold text-white mb-6">Settings</h2>
            
            {/* Noise Reduction */}
            <div className="mb-6">
              <label className="block text-white mb-2">
                Noise Reduction: {noiseReduction}%
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={noiseReduction}
                onChange={(e) => setNoiseReduction(Number(e.target.value))}
                className="w-full"
              />
              <p className="text-blue-200 text-sm mt-1">Higher values filter out more quiet sounds</p>
            </div>
            
            {/* Playback Delay */}
            <div className="mb-6">
              <label className="block text-white mb-2">
                Playback Delay: {playbackDelay}ms
              </label>
              <input
                type="range"
                min="0"
                max="2000"
                step="100"
                value={playbackDelay}
                onChange={(e) => setPlaybackDelay(Number(e.target.value))}
                className="w-full"
              />
              <p className="text-blue-200 text-sm mt-1">Time between capture and playback</p>
            </div>
            
            <div className="bg-blue-500/20 border border-blue-400/30 rounded-lg p-4">
              <p className="text-blue-100 text-sm">
                <strong>How it works:</strong> The system continuously monitors for your voice. 
                When it detects speech, it records for 2 seconds, processes the audio to remove 
                noise, and plays it back to you. This lets you hear yourself as others hear you.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}