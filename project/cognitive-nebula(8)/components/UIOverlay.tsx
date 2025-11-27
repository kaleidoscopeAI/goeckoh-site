import React, { useRef, useEffect, useState } from 'react';
import type { Metrics, ChatMessage, Settings } from '../types';

const LoadingSpinner: React.FC = () => (
  <svg 
    className="animate-spin h-5 w-5 text-cyan-400" 
    xmlns="http://www.w3.org/2000/svg" 
    fill="none" 
    viewBox="0 0 24 24"
  >
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
);

const ProgressBar: React.FC<{ value: number }> = ({ value }) => (
  <div className="w-full bg-cyan-900/50 rounded-full h-2.5">
    <div 
      className="bg-cyan-400 h-2.5 rounded-full transition-all duration-500 ease-out" 
      style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
    ></div>
  </div>
);

const SettingsPanel: React.FC<{ settings: Settings, onSettingsChange: (newSettings: Partial<Settings>) => void, onClose: () => void }> = ({ settings, onSettingsChange, onClose }) => {
    return (
        <div className="absolute inset-0 bg-black/70 backdrop-blur-lg z-50 flex justify-center items-center" onClick={onClose}>
            <div className="bg-gray-900/80 border border-cyan-500/30 rounded-lg p-6 w-full max-w-md text-white font-mono shadow-2xl space-y-4" onClick={e => e.stopPropagation()}>
                <div className="flex justify-between items-center border-b border-cyan-500/30 pb-2 mb-4">
                    <h3 className="font-bold text-cyan-300 text-lg">Settings</h3>
                    <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">&times;</button>
                </div>
                
                <div className="space-y-2">
                    <label className="text-cyan-400 block text-sm">Particle Count</label>
                    <input type="range" min="0.05" max="1" step="0.05" value={settings.particleCount}
                           onChange={e => onSettingsChange({ particleCount: parseFloat(e.target.value) })}
                           className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-400" />
                </div>

                <div className="space-y-2">
                    <label className="text-cyan-400 block text-sm">Color Saturation</label>
                    <input type="range" min="0" max="1" step="0.05" value={settings.colorSaturation}
                           onChange={e => onSettingsChange({ colorSaturation: parseFloat(e.target.value) })}
                           className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-400" />
                </div>

                <div className="space-y-2">
                    <label className="text-cyan-400 block text-sm">Movement Speed</label>
                    <input type="range" min="0.1" max="2" step="0.1" value={settings.movementSpeed}
                           onChange={e => onSettingsChange({ movementSpeed: parseFloat(e.target.value) })}
                           className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-400" />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                    <div className="space-y-2">
                        <label className="text-cyan-400 block text-sm">Temperature</label>
                        <input type="range" min="0.6" max="1.6" step="0.05" value={settings.temperature}
                            onChange={e => onSettingsChange({ temperature: parseFloat(e.target.value) })}
                            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-400" />
                    </div>
                    <div className="space-y-2">
                        <label className="text-cyan-400 block text-sm">Top P</label>
                        <input type="range" min="0.4" max="1" step="0.05" value={settings.topP}
                            onChange={e => onSettingsChange({ topP: parseFloat(e.target.value) })}
                            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-400" />
                    </div>
                    <div className="space-y-2">
                        <label className="text-cyan-400 block text-sm">Presence Penalty</label>
                        <input type="range" min="0" max="1" step="0.05" value={settings.presencePenalty}
                            onChange={e => onSettingsChange({ presencePenalty: parseFloat(e.target.value) })}
                            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-400" />
                    </div>
                    <div className="space-y-2">
                        <label className="text-cyan-400 block text-sm">Style Lens</label>
                        <select
                            value={settings.stylePreset}
                            onChange={e => onSettingsChange({ stylePreset: e.target.value })}
                            className="w-full bg-gray-800 border border-cyan-500/40 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-400 text-white"
                        >
                            <option value="rotate">Rotate styles</option>
                            <option value="cosmic-watercolor">Cosmic watercolor</option>
                            <option value="infrared-neon-noir">Infrared neon noir</option>
                            <option value="bioluminescent-micro">Bioluminescent micro</option>
                            <option value="mythic-architectural">Mythic architectural</option>
                            <option value="subaquatic-dream">Subaquatic dream</option>
                        </select>
                    </div>
                </div>

                <div className="border-t border-cyan-500/20 pt-4">
                    <h4 className="text-cyan-300 font-bold text-sm mb-3">Image Lab (Automatic1111)</h4>
                    <div className="space-y-2">
                        <label className="text-cyan-400 block text-sm">Stable Diffusion Host</label>
                        <input
                            type="text"
                            value={settings.sdHost}
                            onChange={e => onSettingsChange({ sdHost: e.target.value })}
                            className="w-full bg-gray-800 border border-cyan-500/40 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-400"
                            placeholder="http://localhost:7860"
                        />
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
                        <div className="space-y-2">
                            <label className="text-cyan-400 block text-sm">Steps</label>
                            <input
                                type="range"
                                min="10"
                                max="50"
                                step="1"
                                value={settings.sdSteps}
                                onChange={e => onSettingsChange({ sdSteps: parseInt(e.target.value, 10) })}
                                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                            />
                        </div>
                        <div className="space-y-2">
                            <label className="text-cyan-400 block text-sm">CFG Scale</label>
                            <input
                                type="range"
                                min="3"
                                max="15"
                                step="0.5"
                                value={settings.sdCfgScale}
                                onChange={e => onSettingsChange({ sdCfgScale: parseFloat(e.target.value) })}
                                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                            />
                        </div>
                        <div className="space-y-2">
                            <label className="text-cyan-400 block text-sm">Sampler</label>
                            <input
                                type="text"
                                value={settings.sdSampler}
                                onChange={e => onSettingsChange({ sdSampler: e.target.value })}
                                className="w-full bg-gray-800 border border-cyan-500/40 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-400"
                                placeholder="Euler a"
                            />
                        </div>
                        <div className="space-y-2">
                            <label className="text-cyan-400 block text-sm">Seed (blank = random)</label>
                            <input
                                type="number"
                                value={settings.sdSeed ?? ''}
                                onChange={e => {
                                    const value = e.target.value;
                                    onSettingsChange({ sdSeed: value === '' ? null : parseInt(value, 10) });
                                }}
                                className="w-full bg-gray-800 border border-cyan-500/40 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-400"
                                placeholder="random"
                            />
                        </div>
                    </div>
                </div>

                <div className="flex items-center justify-between pt-4">
                    <label className="text-cyan-400 text-sm">Background Stars</label>
                    <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" checked={settings.showStars} onChange={e => onSettingsChange({ showStars: e.target.checked })} className="sr-only peer" />
                        <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cyan-600"></div>
                    </label>
                </div>
                 <div className="flex items-center justify-between">
                    <label className="text-cyan-400 text-sm">Particle Trails</label>
                    <label className="relative inline-flex items-center cursor-pointer">
                        <input type="checkbox" checked={settings.showTrails} onChange={e => onSettingsChange({ showTrails: e.target.checked })} className="sr-only peer" />
                        <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cyan-600"></div>
                    </label>
                </div>

            </div>
        </div>
    );
};


interface UIOverlayProps {
  metrics: Metrics;
  aiThought: string;
  isLoading: boolean;
  isDreaming: boolean;
  onPromptSubmit: (prompt: string) => void;
  chatHistory: ChatMessage[];
  settings: Settings;
  onSettingsChange: (newSettings: Partial<Settings>) => void;
}

const UIOverlay: React.FC<UIOverlayProps> = ({ metrics, aiThought, isLoading, isDreaming, onPromptSubmit, chatHistory, settings, onSettingsChange }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const chatHistoryRef = useRef<HTMLDivElement>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  useEffect(() => {
    if (chatHistoryRef.current) {
        chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [chatHistory]);

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
  
  const getPlaceholderText = () => {
    if (isLoading) {
        return isDreaming ? "Dreaming..." : "Thinking...";
    }
    return "Talk to the nebula...";
  }

  return (
    <>
      <div className="absolute top-4 left-4 md:top-6 md:left-6 flex items-start gap-3">
        <div className="bg-black/50 backdrop-blur-sm p-4 rounded-lg text-white font-mono text-xs md:text-sm space-y-2 shadow-lg border border-cyan-500/20 max-w-sm md:max-w-md">
          <h2 className="font-bold text-cyan-300 text-sm md:text-base border-b border-cyan-500/30 pb-1 mb-2">Cognitive State</h2>
          <div className="flex items-center space-x-2">
            <span className="text-cyan-400 w-24 flex-shrink-0">🧠 Thought:</span> 
            <span className="flex-1 truncate" title={aiThought}>{aiThought || '...'}</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-cyan-400 w-24 flex-shrink-0">🌀 Coherence:</span>
            <ProgressBar value={metrics.coherence} />
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-cyan-400 w-24 flex-shrink-0">❤️ Valence:</span>
            <ProgressBar value={metrics.valence} />
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-cyan-400 w-24 flex-shrink-0">💡 Curiosity:</span>
            <ProgressBar value={metrics.curiosity} />
          </div>
        </div>
        <button onClick={() => setIsSettingsOpen(true)} className="bg-black/50 backdrop-blur-sm p-2 rounded-full text-cyan-300 hover:bg-cyan-900/70 border border-cyan-500/20 transition-colors" aria-label="Open settings">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
        </button>
      </div>
      
      {isSettingsOpen && <SettingsPanel settings={settings} onSettingsChange={onSettingsChange} onClose={() => setIsSettingsOpen(false)} />}
      
      <div className="absolute top-4 right-4 md:top-6 md:right-6 w-[90%] max-w-md h-[calc(100vh-5rem)] flex flex-col bg-black/60 backdrop-blur-md rounded-lg shadow-lg border border-cyan-500/20 text-white">
        <div ref={chatHistoryRef} className="flex-grow p-4 space-y-4 overflow-y-auto font-sans">
            {chatHistory.map((msg, index) => (
                <div key={index} className={`flex items-start gap-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    {msg.role === 'dream' && <span className="opacity-70 text-lg">🌙</span>}
                    <div className={`rounded-lg px-3 py-2 max-w-[85%] text-sm md:text-base ${
                        msg.role === 'user' ? 'bg-cyan-800/70' :
                        msg.role === 'dream' ? 'bg-indigo-900/60' : 'bg-gray-700/60'
                    }`}>
                        <p className={`${msg.role === 'dream' ? 'italic opacity-90' : ''}`} style={{ wordWrap: 'break-word' }}>
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
                    placeholder={getPlaceholderText()}
                    className="w-full bg-black/60 backdrop-blur-sm text-white border border-cyan-400/50 rounded-lg p-3 pr-10 focus:ring-2 focus:ring-cyan-400 focus:border-cyan-400 transition-all duration-300 outline-none placeholder:text-gray-400 disabled:opacity-50 font-sans"
                    onKeyDown={handleKeyDown}
                    disabled={isLoading}
                    aria-label="Chat with AI"
                />
                {isLoading && (
                    <div className="absolute right-3 top-1/2 -translate-y-1/2">
                    <LoadingSpinner />
                    </div>
                )}
            </div>
        </div>
      </div>
    </>
  );
};

export default UIOverlay;
