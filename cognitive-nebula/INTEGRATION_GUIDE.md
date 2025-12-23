# Integration Guide

## 🚀 Complete Integration Guide

This guide shows you how to integrate all the components, hooks, and utilities into your Bubble application.

## 📦 Setup

### 1. Install Dependencies

```bash
cd cognitive-nebula
npm install
```

### 2. Import Styles

Add to your main CSS file:

```css
@import './styles/animations.css';
```

### 3. Setup App Structure

```tsx
import React from 'react';
import { ThemeProvider } from './components/ThemeProvider';
import ErrorBoundary from './components/ErrorBoundary';
import MobileOptimized from './components/MobileOptimized';
import { useVoiceData } from './hooks/useVoiceData';
import { useSession } from './hooks/useSession';
import { useBubbleState } from './hooks/useBubbleState';
import EnhancedThreeCanvas from './components/EnhancedThreeCanvas';
import EnhancedUIOverlay from './components/EnhancedUIOverlay';
```

## 🔌 Voice Pipeline Integration

### Option 1: Using WebSocket (Recommended)

```tsx
import { connectVoicePipeline } from './utils/voicePipeline';

const App = () => {
  const { voiceData, isConnected } = useVoiceData({
    updateInterval: 100,
    onUpdate: (data) => {
      console.log('Voice data updated', data);
    }
  });

  useEffect(() => {
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
  }, []);
};
```

### Option 2: Using Callbacks

```tsx
// In your voice processing code
const handleVoiceData = (audioBuffer: Float32Array) => {
  extractVoiceMetrics(audioBuffer).then(metrics => {
    setVoiceMetrics(metrics);
  });
};
```

## 🎯 Complete App Example

```tsx
import React, { useState, useEffect } from 'react';
import { ThemeProvider } from './components/ThemeProvider';
import ErrorBoundary from './components/ErrorBoundary';
import MobileOptimized from './components/MobileOptimized';
import { useVoiceData } from './hooks/useVoiceData';
import { useSession } from './hooks/useSession';
import { useBubbleState } from './hooks/useBubbleState';
import EnhancedThreeCanvas from './components/EnhancedThreeCanvas';
import EnhancedUIOverlay from './components/EnhancedUIOverlay';
import PerformanceMonitor from './components/PerformanceMonitor';

const AppContent = () => {
  // Voice data hook
  const { voiceData, isConnected } = useVoiceData({
    updateInterval: 100
  });

  // Session hook
  const {
    currentSession,
    isActive,
    startSession,
    endSession,
    completeExercise,
    updateMetrics,
    getSessionTime
  } = useSession({
    autoStart: false
  });

  // Bubble state hook
  const { bubbleState } = useBubbleState(
    voiceData ? {
      energy: voiceData.metrics.energy,
      f0: voiceData.metrics.f0,
      zcr: 0.3, // Calculate from audio
      tilt: 0.5,
      hnr: 0.8
    } : null
  );

  // Update session metrics
  useEffect(() => {
    if (voiceData && isActive) {
      updateMetrics(
        voiceData.metrics.clarity,
        voiceData.metrics.fluency
      );
    }
  }, [voiceData, isActive, updateMetrics]);

  return (
    <div className="relative w-full h-screen bg-black overflow-hidden">
      {/* 3D Visualization */}
      <EnhancedThreeCanvas
        metrics={metrics}
        aiThought={aiThought}
        imageData={imageData}
        settings={settings}
        voiceData={{
          bubbleState,
          waveform: voiceData?.waveform,
          spectrum: voiceData?.spectrum
        }}
      />

      {/* UI Overlay */}
      <EnhancedUIOverlay
        metrics={metrics}
        aiThought={aiThought}
        isLoading={isLoading}
        isDreaming={isDreaming}
        onPromptSubmit={handlePromptSubmit}
        chatHistory={chatHistory}
        settings={settings}
        onSettingsChange={handleSettingsChange}
        voiceData={voiceData ? {
          energy: voiceData.metrics.energy,
          f0: voiceData.metrics.f0,
          clarity: voiceData.metrics.clarity,
          fluency: voiceData.metrics.fluency,
          volume: voiceData.metrics.volume,
          waveform: voiceData.waveform,
          spectrum: voiceData.spectrum
        } : undefined}
        showVoicePanel={true}
        showProgress={true}
        showVoiceFeedback={true}
      />

      {/* Performance Monitor (dev only) */}
      {process.env.NODE_ENV === 'development' && (
        <PerformanceMonitor enabled={true} />
      )}
    </div>
  );
};

const App = () => {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <MobileOptimized>
          <AppContent />
        </MobileOptimized>
      </ThemeProvider>
    </ErrorBoundary>
  );
};

export default App;
```

## 🔗 Connecting to Your Voice Pipeline

### Step 1: Update useVoiceData Hook

Replace the mock connection in `hooks/useVoiceData.ts`:

```tsx
// Replace this:
const connectToVoicePipeline = () => {
  // Mock connection
};

// With your actual connection:
const connectToVoicePipeline = () => {
  // Your WebSocket or callback connection
  const ws = new WebSocket('ws://localhost:8765');
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // Update voice data
    setVoiceData(processVoiceData(data));
  };
  
  return () => ws.close();
};
```

### Step 2: Map Your Voice Data

```tsx
const processVoiceData = (rawData: any) => {
  return {
    metrics: {
      energy: rawData.energy || 0,
      f0: rawData.f0 || 220,
      clarity: calculateClarity(rawData),
      fluency: calculateFluency(rawData),
      volume: rawData.volume || 0,
      pitchStability: calculatePitchStability(rawData)
    },
    waveform: rawData.waveform || new Float32Array(512),
    spectrum: rawData.spectrum || new Float32Array(128),
    timestamp: Date.now()
  };
};
```

## 🎨 Customization

### Custom Theme

```tsx
<ThemeProvider defaultTheme="dark" defaultColorScheme="purple">
  <YourApp />
</ThemeProvider>
```

### Custom Preferences

```tsx
import UserPreferences, { defaultPreferences } from './components/UserPreferences';

const [prefs, setPrefs] = useState(defaultPreferences);

<UserPreferences
  preferences={prefs}
  onPreferencesChange={(newPrefs) => setPrefs({ ...prefs, ...newPrefs })}
/>
```

## 📊 Data Flow

```
Voice Pipeline
    ↓
useVoiceData Hook
    ↓
useBubbleState Hook
    ↓
EnhancedThreeCanvas (Visual)
    ↓
EnhancedUIOverlay (UI)
    ↓
VoiceTherapyPanel (Feedback)
```

## 🧪 Testing

### Mock Voice Data

```tsx
const mockVoiceData = {
  metrics: {
    energy: 0.7,
    f0: 220,
    clarity: 0.85,
    fluency: 0.78,
    volume: 0.6,
    pitchStability: 0.8
  },
  waveform: new Float32Array(512).map(() => Math.random() * 2 - 1),
  spectrum: new Float32Array(128).map(() => Math.random())
};
```

## 🐛 Troubleshooting

### Voice Data Not Updating

1. Check WebSocket connection
2. Verify data format
3. Check console for errors
4. Verify hook dependencies

### Performance Issues

1. Enable PerformanceMonitor
2. Reduce update frequency
3. Check particle counts
4. Disable heavy effects

### Mobile Issues

1. Test on real device
2. Check touch targets
3. Verify viewport meta
4. Test gestures

## 📚 Next Steps

1. Connect your voice pipeline
2. Customize themes and preferences
3. Add your own components
4. Test on all devices
5. Deploy!

---

**Ready to integrate!** Follow this guide step by step. 🚀

