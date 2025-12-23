import React, { useState, useRef, useEffect } from 'react';
import type { Metrics, ChatMessage, Settings } from '../types';
import VoiceTherapyPanel from './VoiceTherapyPanel';
import ProgressDashboard from './ProgressDashboard';
import RealTimeVoiceFeedback from './RealTimeVoiceFeedback';

interface EnhancedUIOverlayProps {
  metrics: Metrics;
  aiThought: string;
  isLoading: boolean;
  isDreaming: boolean;
  onPromptSubmit: (prompt: string) => void;
  chatHistory: ChatMessage[];
  settings: Settings;
  onSettingsChange: (newSettings: Partial<Settings>) => void;
  voiceData?: {
    energy: number;
    f0: number;
    clarity: number;
    fluency: number;
    volume: number;
    waveform?: Float32Array;
    spectrum?: Float32Array;
  };
  showVoicePanel?: boolean;
  showProgress?: boolean;
  showVoiceFeedback?: boolean;
}

/**
 * Enhanced UI Overlay with voice therapy features
 * Combines chat, metrics, voice feedback, and progress tracking
 */
const EnhancedUIOverlay: React.FC<EnhancedUIOverlayProps> = ({
  metrics,
  aiThought,
  isLoading,
  isDreaming,
  onPromptSubmit,
  chatHistory,
  settings,
  onSettingsChange,
  voiceData,
  showVoicePanel = true,
  showProgress = false,
  showVoiceFeedback = true
}) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const chatHistoryRef = useRef<HTMLDivElement>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isVoicePanelOpen, setIsVoicePanelOpen] = useState(showVoicePanel);
  const [isProgressOpen, setIsProgressOpen] = useState(showProgress);
  const [sessionTime, setSessionTime] = useState(0);
  const [exercisesCompleted, setExercisesCompleted] = useState(0);
  const [isActive, setIsActive] = useState(false);

  useEffect(() => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [chatHistory]);

  useEffect(() => {
    if (!isActive) return;
    const interval = setInterval(() => {
      setSessionTime(prev => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [isActive]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      const inputElement = inputRef.current;
      if (inputElement) {
        const query = inputElement.value.trim();
        if (query.length > 0 && !isLoading) {
          onPromptSubmit(query);
          inputElement.value = '';
        }
      }
    }
  };

  const handleStartExercise = () => {
    setIsActive(true);
    setExercisesCompleted(prev => prev + 1);
  };

  const handlePause = () => {
    setIsActive(false);
  };

  return (
    <>
      {/* Main Metrics Panel */}
      <div className="absolute top-4 left-4 md:top-6 md:left-6 flex items-start gap-3 z-10">
        <div className="bg-black/70 backdrop-blur-lg p-4 rounded-xl text-white font-mono text-xs md:text-sm space-y-2 shadow-2xl border border-cyan-500/30 max-w-sm md:max-w-md">
          <div className="flex items-center justify-between mb-2">
            <h2 className="font-bold text-cyan-300 text-sm md:text-base">Cognitive State</h2>
            <div className="flex gap-2">
              <button
                onClick={() => setIsVoicePanelOpen(!isVoicePanelOpen)}
                className={`p-1 rounded ${isVoicePanelOpen ? 'bg-cyan-600' : 'bg-gray-700'} hover:bg-cyan-700 transition-colors`}
                title="Toggle Voice Panel"
              >
                🎤
              </button>
              <button
                onClick={() => setIsProgressOpen(!isProgressOpen)}
                className={`p-1 rounded ${isProgressOpen ? 'bg-cyan-600' : 'bg-gray-700'} hover:bg-cyan-700 transition-colors`}
                title="Toggle Progress"
              >
                📊
              </button>
              <button
                onClick={() => setIsSettingsOpen(true)}
                className="p-1 rounded bg-gray-700 hover:bg-cyan-700 transition-colors"
                title="Settings"
              >
                ⚙️
              </button>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="text-cyan-400 w-24 flex-shrink-0">🧠 Thought:</span>
            <span className="flex-1 truncate" title={aiThought}>{aiThought || '...'}</span>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <span className="text-cyan-400 w-24 flex-shrink-0">🌀 Coherence:</span>
              <div className="flex-1 bg-gray-800 rounded-full h-2">
                <div
                  className="bg-cyan-400 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${metrics.coherence * 100}%` }}
                />
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-cyan-400 w-24 flex-shrink-0">❤️ Valence:</span>
              <div className="flex-1 bg-gray-800 rounded-full h-2">
                <div
                  className="bg-pink-400 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${metrics.valence * 100}%` }}
                />
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-cyan-400 w-24 flex-shrink-0">💡 Curiosity:</span>
              <div className="flex-1 bg-gray-800 rounded-full h-2">
                <div
                  className="bg-yellow-400 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${metrics.curiosity * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Voice Therapy Panel */}
      {isVoicePanelOpen && voiceData && (
        <VoiceTherapyPanel
          voiceMetrics={{
            energy: voiceData.energy,
            f0: voiceData.f0,
            clarity: voiceData.clarity,
            fluency: voiceData.fluency,
            volume: voiceData.volume,
            pitchStability: 0.8 // TODO: Calculate from voice data
          }}
          sessionTime={sessionTime}
          exercisesCompleted={exercisesCompleted}
          onStartExercise={handleStartExercise}
          onPause={handlePause}
          isActive={isActive}
        />
      )}

      {/* Progress Dashboard */}
      {isProgressOpen && (
        <ProgressDashboard
          sessions={[
            {
              date: new Date().toISOString(),
              duration: sessionTime,
              exercisesCompleted,
              averageClarity: voiceData?.clarity || 0.7,
              averageFluency: voiceData?.fluency || 0.7,
              improvements: []
            }
          ]}
          currentStreak={7}
          totalTime={sessionTime}
          onExport={() => console.log('Export progress')}
        />
      )}

      {/* Real-time Voice Feedback */}
      {showVoiceFeedback && voiceData && voiceData.waveform && voiceData.spectrum && (
        <RealTimeVoiceFeedback
          voiceData={{
            waveform: voiceData.waveform,
            spectrum: voiceData.spectrum,
            f0: voiceData.f0,
            energy: voiceData.energy,
            clarity: voiceData.clarity,
            volume: voiceData.volume
          }}
          size="medium"
          position="top-right"
        />
      )}

      {/* Chat Interface */}
      <div className="absolute top-4 right-4 md:top-6 md:right-6 w-[90%] max-w-md h-[calc(100vh-5rem)] flex flex-col bg-black/70 backdrop-blur-lg rounded-xl shadow-2xl border border-cyan-500/30 text-white z-10">
        <div ref={chatHistoryRef} className="flex-grow p-4 space-y-4 overflow-y-auto font-sans">
          {chatHistory.map((msg, index) => (
            <div key={index} className={`flex items-start gap-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.role === 'dream' && <span className="opacity-70 text-lg">🌙</span>}
              <div className={`rounded-lg px-3 py-2 max-w-[85%] text-sm md:text-base ${
                msg.role === 'user' ? 'bg-cyan-800/70' :
                msg.role === 'dream' ? 'bg-indigo-900/60' : 'bg-gray-700/60'
              }`}>
                <p className={msg.role === 'dream' ? 'italic opacity-90' : ''} style={{ wordWrap: 'break-word' }}>
                  {msg.content}
                </p>
              </div>
            </div>
          ))}
        </div>
        <div className="p-2 md:p-4 border-t border-cyan-500/20">
          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              placeholder={isLoading ? (isDreaming ? "Dreaming..." : "Thinking...") : "Talk to the nebula..."}
              className="w-full bg-black/60 backdrop-blur-sm text-white border border-cyan-400/50 rounded-lg p-3 pr-10 focus:ring-2 focus:ring-cyan-400 focus:border-cyan-400 transition-all duration-300 outline-none placeholder:text-gray-400 disabled:opacity-50 font-sans"
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              aria-label="Chat with AI"
            />
            {isLoading && (
              <div className="absolute right-3 top-1/2 -translate-y-1/2">
                <div className="animate-spin h-5 w-5 border-2 border-cyan-400 border-t-transparent rounded-full" />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Settings Panel (reuse existing) */}
      {/* TODO: Import SettingsPanel component */}
    </>
  );
};

export default EnhancedUIOverlay;

