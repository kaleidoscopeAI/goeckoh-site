/**
 * Complete Enhanced App Example
 * 
 * This demonstrates how to integrate all the new components
 * into a production-ready application.
 */

import React, { useState, useEffect } from 'react';
import { ThemeProvider, ThemeToggle, useTheme } from '../components/ThemeProvider';
import ErrorBoundary from '../components/ErrorBoundary';
import MobileOptimized from '../components/MobileOptimized';
import PerformanceMonitor from '../components/PerformanceMonitor';
import { useKeyboardShortcuts, KeyboardShortcuts } from '../components/KeyboardShortcuts';
import LoadingStates from '../components/LoadingStates';
import EnhancedUIOverlay from '../components/EnhancedUIOverlay';
import EnhancedThreeCanvas from '../components/EnhancedThreeCanvas';
import type { Metrics, Settings, ChatMessage } from '../types';

// Mock data - replace with real data from your voice pipeline
const initialMetrics: Metrics = {
  curiosity: 0.7,
  confusion: 0.3,
  coherence: 0.5,
  arousal: 0.4,
  valence: 0.6,
  dominance: 0.5,
  certainty: 0.5,
  resonance: 0.6,
};

const defaultSettings: Settings = {
  particleCount: 0.5,
  colorSaturation: 1.0,
  movementSpeed: 1.0,
  showStars: true,
  showTrails: true,
  temperature: 1.1,
  topP: 0.9,
  presencePenalty: 0.4,
  stylePreset: 'rotate',
  sdHost: 'http://localhost:7860',
  sdSteps: 24,
  sdCfgScale: 7.5,
  sdSampler: 'Euler a',
  sdSeed: null,
};

const AppContent: React.FC = () => {
  const [metrics, setMetrics] = useState<Metrics>(initialMetrics);
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isDreaming, setIsDreaming] = useState(false);
  const [aiThought, setAiThought] = useState('A silent nebula awaits a spark of inquiry.');
  const [imageData, setImageData] = useState<string | null>(null);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [voiceData, setVoiceData] = useState({
    energy: 0.5,
    f0: 220,
    clarity: 0.7,
    fluency: 0.7,
    volume: 0.5,
    waveform: new Float32Array(512),
    spectrum: new Float32Array(128)
  });

  const { theme } = useTheme();

  // Keyboard shortcuts
  const shortcuts = [
    {
      key: 'k',
      description: 'Focus chat input',
      action: () => {
        const input = document.querySelector('input[type="text"]') as HTMLInputElement;
        input?.focus();
      }
    },
    {
      key: '?',
      description: 'Show keyboard shortcuts',
      action: () => setShowShortcuts(!showShortcuts)
    },
    {
      key: 's',
      modifier: 'ctrl' as const,
      description: 'Open settings',
      action: () => {
        // Open settings panel
        console.log('Open settings');
      }
    },
    {
      key: 'Escape',
      description: 'Close panels',
      action: () => {
        setShowShortcuts(false);
      }
    }
  ];

  useKeyboardShortcuts(shortcuts);

  // Simulate voice data updates (replace with real voice pipeline)
  useEffect(() => {
    const interval = setInterval(() => {
      setVoiceData(prev => ({
        ...prev,
        energy: Math.max(0, Math.min(1, prev.energy + (Math.random() - 0.5) * 0.1)),
        f0: 200 + Math.random() * 100,
        clarity: Math.max(0, Math.min(1, prev.clarity + (Math.random() - 0.5) * 0.05)),
        fluency: Math.max(0, Math.min(1, prev.fluency + (Math.random() - 0.5) * 0.05)),
        volume: Math.max(0, Math.min(1, prev.volume + (Math.random() - 0.5) * 0.1)),
        waveform: new Float32Array(512).map(() => Math.random() * 2 - 1),
        spectrum: new Float32Array(128).map(() => Math.random())
      }));
    }, 100);

    return () => clearInterval(interval);
  }, []);

  const handlePromptSubmit = async (prompt: string) => {
    setIsLoading(true);
    setChatHistory(prev => [...prev, { role: 'user', content: prompt }]);
    
    // Simulate AI response
    setTimeout(() => {
      setAiThought(`Processing: ${prompt}`);
      setChatHistory(prev => [...prev, { role: 'model', content: `Response to: ${prompt}` }]);
      setIsLoading(false);
    }, 1000);
  };

  const handleSettingsChange = (newSettings: Partial<Settings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  };

  return (
    <div className="relative w-full h-screen bg-black overflow-hidden">
      {/* Main 3D Visualization */}
      <EnhancedThreeCanvas
        metrics={metrics}
        aiThought={aiThought}
        imageData={imageData}
        settings={settings}
        voiceData={{
          bubbleState: {
            radius: 1.0 + voiceData.energy * 0.5,
            color_r: 0.25,
            color_g: 0.7,
            color_b: 1.0,
            rough: 0.4,
            metal: 0.6,
            spike: 0.3,
            energy: voiceData.energy,
            f0: voiceData.f0
          },
          waveform: voiceData.waveform,
          spectrum: voiceData.spectrum
        }}
      />

      {/* Enhanced UI Overlay */}
      <EnhancedUIOverlay
        metrics={metrics}
        aiThought={aiThought}
        isLoading={isLoading}
        isDreaming={isDreaming}
        onPromptSubmit={handlePromptSubmit}
        chatHistory={chatHistory}
        settings={settings}
        onSettingsChange={handleSettingsChange}
        voiceData={voiceData}
        showVoicePanel={true}
        showProgress={false}
        showVoiceFeedback={true}
      />

      {/* Keyboard Shortcuts Help */}
      <KeyboardShortcuts
        shortcuts={shortcuts}
        showHelp={showShortcuts}
        onToggleHelp={() => setShowShortcuts(!showShortcuts)}
      />

      {/* Theme Toggle (floating) */}
      <div className="fixed bottom-4 right-4 z-50">
        <ThemeToggle />
      </div>

      {/* Performance Monitor (dev mode) */}
      {process.env.NODE_ENV === 'development' && (
        <PerformanceMonitor
          enabled={true}
          position="top-right"
          showDetails={true}
        />
      )}

      {/* Loading Overlay */}
      {isLoading && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <LoadingStates
            type="spinner"
            message="Processing your request..."
            size="large"
          />
        </div>
      )}
    </div>
  );
};

const CompleteEnhancedApp: React.FC = () => {
  return (
    <ErrorBoundary
      onError={(error, errorInfo) => {
        console.error('App Error:', error, errorInfo);
        // Send to error tracking service
      }}
    >
      <ThemeProvider defaultTheme="dark" defaultColorScheme="cyan">
        <MobileOptimized breakpoint={768}>
          <AppContent />
        </MobileOptimized>
      </ThemeProvider>
    </ErrorBoundary>
  );
};

export default CompleteEnhancedApp;

