import React, { useState, useCallback, useEffect, useRef } from 'react';
import ThreeCanvas from './components/ThreeCanvas';
import UIOverlay from './components/UIOverlay';
import { createChatSession, sendMessage, generateImageFromThought, findAndDescribeImageOnWeb } from './services/aiService';
import type { Metrics, ChatMessage, Settings } from './types';
import type { ChatSession } from './services/aiService';

const INITIAL_METRICS: Metrics = {
    curiosity: 0.7,
    confusion: 0.3,
    coherence: 0.5,
    arousal: 0.4,
    valence: 0.6,
    dominance: 0.5,
    certainty: 0.5,
    resonance: 0.6,
};

const DEFAULT_SETTINGS: Settings = {
    particleCount: 0.5, // Corresponds to 15000 nodes
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

const INITIAL_THOUGHT = 'A silent nebula awaits a spark of inquiry.';
const IDLE_TIMEOUT = 20000; // 20 seconds
const MAX_IDLE_TIMEOUT = 300000; // 5 minutes

interface ImagePaletteMetrics {
    avgLightness: number;
    avgSaturation: number;
}

const rgbToHsl = (r: number, g: number, b: number): [number, number, number] => {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h = 0, s = 0, l = (max + min) / 2;
    if (max !== min) {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }
    return [h, s, l];
};

const analyzeImagePalette = (b64Image: string): Promise<ImagePaletteMetrics> => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const downscaleFactor = Math.max(1, Math.floor(img.width / 128));
            const width = img.width / downscaleFactor;
            const height = img.height / downscaleFactor;
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            if (!ctx) return reject(new Error("Could not get canvas context"));

            ctx.drawImage(img, 0, 0, width, height);
            const imageData = ctx.getImageData(0, 0, width, height);
            const pixels = imageData.data;

            let totalLightness = 0;
            let totalSaturation = 0;
            let pixelCount = 0;
            const step = 4; // Sample every pixel

            for (let i = 0; i < pixels.length; i += 4 * step) {
                const r = pixels[i];
                const g = pixels[i + 1];
                const b = pixels[i + 2];
                const [, s, l] = rgbToHsl(r, g, b);
                totalSaturation += s;
                totalLightness += l;
                pixelCount++;
            }

            resolve({
                avgLightness: pixelCount > 0 ? totalLightness / pixelCount : 0.5,
                avgSaturation: pixelCount > 0 ? totalSaturation / pixelCount : 0.5,
            });
        };
        img.onerror = () => reject(new Error("Image failed to load for analysis."));
        img.src = `data:image/png;base64,${b64Image}`;
    });
};


const calculateMetricsFromThought = (thought: string, imageMetrics?: ImagePaletteMetrics): Metrics => {
    const lowerThought = thought.toLowerCase();
    const words = lowerThought.split(/\s+/);
    const wordCount = words.length;

    const positiveWords = ['beautiful', 'vibrant', 'joy', 'love', 'happy', 'bright', 'stunning', 'wonderful', 'serene', 'peaceful'];
    const negativeWords = ['dark', 'sad', 'empty', 'lonely', 'fear', 'stormy', 'gloomy', 'error', 'fail'];
    let textValence = 0.5;
    if (positiveWords.some(word => lowerThought.includes(word))) textValence += 0.3;
    if (negativeWords.some(word => lowerThought.includes(word))) textValence -= 0.3;

    const highArousalWords = ['explosion', 'fast', 'racing', 'vibrant', 'intense', 'epic', 'dynamic', 'suddenly'];
    const lowArousalWords = ['calm', 'serene', 'peaceful', 'slow', 'gentle', 'tranquil', 'quiet'];
    let textArousal = 0.4;
    if (highArousalWords.some(word => lowerThought.includes(word))) textArousal += 0.4;
    if (lowArousalWords.some(word => lowerThought.includes(word))) textArousal -= 0.2;

    let certaintyScore = 0.4 + (Math.min(wordCount, 50) / 50) * 0.5;
    let coherenceScore = 0.5 + (certaintyScore - 0.4) * 0.8;
    let resonanceScore = 0.4 + (Math.min(wordCount, 50) / 50) * 0.4;
    let curiosityScore = Math.random() * 0.3 + 0.2;
    if (lowerThought.includes('?')) curiosityScore += 0.3;
    let dominanceScore = 0.3 + certaintyScore * 0.4;

    let finalValence = textValence;
    let finalArousal = textArousal;

    if (imageMetrics) {
        // Blend text-based metrics with image-based metrics
        finalValence = (textValence * 0.6) + (imageMetrics.avgLightness * 0.4);
        finalArousal = (textArousal * 0.5) + (imageMetrics.avgSaturation * 0.5);
    }
    
    const clamp = (val: number) => Math.max(0.05, Math.min(0.95, val));

    return {
        curiosity: clamp(curiosityScore),
        coherence: clamp(coherenceScore),
        valence: clamp(finalValence),
        certainty: clamp(certaintyScore),
        arousal: clamp(finalArousal),
        resonance: clamp(resonanceScore),
        dominance: clamp(dominanceScore),
        confusion: clamp(1 - certaintyScore),
    };
};

function App() {
  const [metrics, setMetrics] = useState<Metrics>(INITIAL_METRICS);
  const [targetMetrics, setTargetMetrics] = useState<Metrics | null>(null);
  const [aiThought, setAiThought] = useState<string>(INITIAL_THOUGHT);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isDreaming, setIsDreaming] = useState<boolean>(false);
  const [imageData, setImageData] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  
  const chatRef = useRef<ChatSession | null>(null);
  const metricsRef = useRef(metrics);
  const idleTimerRef = useRef<number>();
  const idleTimeoutDurationRef = useRef<number>(IDLE_TIMEOUT);

  // Load settings from localStorage on initial render
  useEffect(() => {
    try {
      const savedSettings = localStorage.getItem('cognitiveNebulaSettings');
      if (savedSettings) {
        setSettings({ ...DEFAULT_SETTINGS, ...JSON.parse(savedSettings) });
      }
    } catch (error) {
      console.error("Failed to load settings from localStorage", error);
    }
  }, []);

  // Save settings to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem('cognitiveNebulaSettings', JSON.stringify(settings));
    } catch (error) {
      console.error("Failed to save settings to localStorage", error);
    }
  }, [settings]);

  const handleSettingsChange = (newSettings: Partial<Settings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  };

  useEffect(() => {
    metricsRef.current = metrics;
  }, [metrics]);

  useEffect(() => {
    chatRef.current = createChatSession();
    setChatHistory([{ role: 'model', content: INITIAL_THOUGHT }]);
  }, []);

  useEffect(() => {
    if (!targetMetrics) return;

    let animationFrameId: number;
    const startMetrics = { ...metricsRef.current };
    const duration = 1200; // ms
    let startTime: number | null = null;

    const animate = (timestamp: number) => {
        if (!startTime) startTime = timestamp;
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easedProgress = 1 - Math.pow(1 - progress, 3); // Ease-out cubic

        const nextMetrics: Partial<Metrics> = {};
        for (const key in targetMetrics) {
            const metricKey = key as keyof Metrics;
            const startValue = startMetrics[metricKey];
            const endValue = targetMetrics[metricKey];
            nextMetrics[metricKey] = startValue + (endValue - startValue) * easedProgress;
        }

        setMetrics(prev => ({ ...prev, ...nextMetrics }));

        if (progress < 1) {
            animationFrameId = requestAnimationFrame(animate);
        } else {
            setTargetMetrics(null); // End animation
        }
    };

    animationFrameId = requestAnimationFrame(animate);

    return () => {
        cancelAnimationFrame(animationFrameId);
    };
  }, [targetMetrics]);

  const handleIdleGeneration = useCallback(async () => {
    if (!chatRef.current) return;
    
    setIsLoading(true);
    setIsDreaming(true);
    setImageData(null);
    setAiThought('Dreaming of something new...');
    setChatHistory(prev => [...prev, { role: 'dream', content: '...' }]);

    const lastMessage = chatHistory[chatHistory.length - 1]?.content || 'something beautiful and abstract';
    const dreamPrompt = `Based on the idea of "${lastMessage}", have a new, creative, and visual thought. Describe it vividly.`;

    try {
        const thought = await sendMessage(chatRef.current, dreamPrompt, settings);
        setAiThought(thought);
        setChatHistory(prev => [...prev.slice(0, -1), { role: 'dream', content: thought }]);

        const imageDescription = await findAndDescribeImageOnWeb(thought, settings, chatRef.current);
        if (chatRef.current) chatRef.current.lastImageDescription = imageDescription;
        const b64Image = await generateImageFromThought(imageDescription, settings, chatRef.current?.lastStylePrompt);
        
        let newMetricsTarget;
        if (b64Image) {
            setImageData(b64Image);
            const imageMetrics = await analyzeImagePalette(b64Image);
            newMetricsTarget = calculateMetricsFromThought(thought, imageMetrics);
        } else {
            newMetricsTarget = calculateMetricsFromThought(thought);
        }
        setTargetMetrics(newMetricsTarget);
        idleTimeoutDurationRef.current = IDLE_TIMEOUT; // Reset backoff on success

    } catch (error) {
        console.error("Error during autonomous generation:", error);
        const errorMessage = String(error);
        setAiThought(errorMessage);
        setChatHistory(prev => [...prev.slice(0, -1), { role: 'dream', content: errorMessage }]);

        if (errorMessage.toLowerCase().includes("exceeded")) {
            const newTimeout = Math.min(idleTimeoutDurationRef.current * 2, MAX_IDLE_TIMEOUT);
            idleTimeoutDurationRef.current = newTimeout;
            console.log(`Rate limit hit. Backing off idle timer to ${newTimeout / 1000}s.`);
        }

        setTargetMetrics({
            curiosity: 0.2, confusion: 0.9, coherence: 0.1, arousal: 0.8, 
            valence: 0.1, dominance: 0.2, certainty: 0.1, resonance: 0.1,
        });
    } finally {
        setIsLoading(false);
        setIsDreaming(false);
    }
  }, [chatHistory, settings]);
  
  useEffect(() => {
    clearTimeout(idleTimerRef.current);
    if (!isLoading) {
        idleTimerRef.current = window.setTimeout(handleIdleGeneration, idleTimeoutDurationRef.current);
    }
    return () => clearTimeout(idleTimerRef.current);
  }, [isLoading, chatHistory, handleIdleGeneration]);


  const handlePromptSubmit = useCallback(async (prompt: string) => {
    if (!chatRef.current) return;

    setIsLoading(true);
    setImageData(null); 
    setChatHistory(prev => [...prev, { role: 'user', content: prompt }]);
    setAiThought('Contemplating the query...');
    
    try {
      const thought = await sendMessage(chatRef.current, prompt, settings);
      setAiThought(thought);
      setChatHistory(prev => [...prev, { role: 'model', content: thought }]);
      
      const imageDescription = await findAndDescribeImageOnWeb(thought, settings, chatRef.current);
      if (chatRef.current) chatRef.current.lastImageDescription = imageDescription;
      const b64Image = await generateImageFromThought(imageDescription, settings, chatRef.current?.lastStylePrompt);
      
      let newMetricsTarget;
      if (b64Image) { 
        setImageData(b64Image);
        try {
            const imageMetrics = await analyzeImagePalette(b64Image);
            newMetricsTarget = calculateMetricsFromThought(thought, imageMetrics);
        } catch(e) {
            console.error("Could not analyze image palette, falling back to text only.", e)
            newMetricsTarget = calculateMetricsFromThought(thought);
        }
      } else {
        newMetricsTarget = calculateMetricsFromThought(thought);
      }
      
      setTargetMetrics(newMetricsTarget);
      idleTimeoutDurationRef.current = IDLE_TIMEOUT; // Reset backoff on success
    } catch (error) {
        console.error("Error processing prompt:", error);
        const errorMessage = String(error);
        setAiThought(`Error: ${errorMessage}`);
        setChatHistory(prev => [...prev, { role: 'model', content: `Error: ${errorMessage}` }]);
        
        if (errorMessage.toLowerCase().includes("exceeded")) {
            const newTimeout = Math.min(idleTimeoutDurationRef.current * 2, MAX_IDLE_TIMEOUT);
            idleTimeoutDurationRef.current = newTimeout;
            console.log(`Rate limit hit. Backing off idle timer to ${newTimeout / 1000}s.`);
        }

        setTargetMetrics({
            curiosity: 0.2, confusion: 0.9, coherence: 0.1, arousal: 0.8, 
            valence: 0.1, dominance: 0.2, certainty: 0.1, resonance: 0.1,
        });
    } finally {
      setIsLoading(false);
    }
  }, [settings]);

  return (
    <main className="w-full h-screen bg-black relative overflow-hidden font-sans">
      <ThreeCanvas metrics={metrics} aiThought={aiThought} imageData={imageData} settings={settings} />
      <UIOverlay 
        metrics={metrics}
        aiThought={aiThought}
        isLoading={isLoading}
        isDreaming={isDreaming}
        onPromptSubmit={handlePromptSubmit}
        chatHistory={chatHistory}
        settings={settings}
        onSettingsChange={handleSettingsChange}
      />
    </main>
  );
}

export default App;
