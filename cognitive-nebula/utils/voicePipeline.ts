/**
 * Voice Pipeline Integration Utilities
 * 
 * Helper functions for connecting to the Bubble voice processing pipeline
 */

export interface VoicePipelineConfig {
  wsUrl?: string;
  updateInterval?: number;
  onVoiceData?: (data: any) => void;
  onError?: (error: Error) => void;
}

/**
 * Connect to voice pipeline via WebSocket
 */
export const connectVoicePipeline = (config: VoicePipelineConfig = {}) => {
  const {
    wsUrl = 'ws://localhost:8765',
    onVoiceData,
    onError
  } = config;

  let ws: WebSocket | null = null;
  let reconnectTimeout: NodeJS.Timeout | null = null;

  const connect = () => {
    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('[Voice Pipeline] Connected');
        if (reconnectTimeout) {
          clearTimeout(reconnectTimeout);
          reconnectTimeout = null;
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onVoiceData?.(data);
        } catch (error) {
          console.error('[Voice Pipeline] Failed to parse message', error);
          onError?.(error as Error);
        }
      };

      ws.onerror = (error) => {
        console.error('[Voice Pipeline] WebSocket error', error);
        onError?.(new Error('WebSocket connection error'));
      };

      ws.onclose = () => {
        console.log('[Voice Pipeline] Disconnected, reconnecting...');
        reconnectTimeout = setTimeout(connect, 2000);
      };
    } catch (error) {
      console.error('[Voice Pipeline] Connection failed', error);
      onError?.(error as Error);
      reconnectTimeout = setTimeout(connect, 2000);
    }
  };

  connect();

  return {
    disconnect: () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (ws) {
        ws.close();
        ws = null;
      }
    },
    send: (data: any) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(data));
      }
    }
  };
};

/**
 * Process voice data from bubble state
 */
export const processBubbleState = (data: {
  radius?: number;
  spike?: number;
  metalness?: number;
  roughness?: number;
  hue?: number;
  halo?: number;
  idle?: boolean;
}) => {
  return {
    bubbleState: {
      radius: data.radius || 1.0,
      color_r: 0.25,
      color_g: 0.7,
      color_b: 1.0,
      rough: data.roughness || 0.4,
      metal: data.metalness || 0.6,
      spike: data.spike || 0.0,
      energy: data.idle ? 0 : 0.5,
      f0: 220
    }
  };
};

/**
 * Extract voice metrics from audio buffer
 */
export const extractVoiceMetrics = async (
  audioBuffer: Float32Array,
  sampleRate: number = 22050
): Promise<{
  energy: number;
  f0: number;
  zcr: number;
  clarity: number;
  fluency: number;
}> => {
  // Calculate RMS energy
  const energy = Math.sqrt(
    audioBuffer.reduce((sum, sample) => sum + sample * sample, 0) / audioBuffer.length
  );

  // Calculate Zero Crossing Rate (simplified)
  let zcr = 0;
  for (let i = 1; i < audioBuffer.length; i++) {
    if ((audioBuffer[i] >= 0) !== (audioBuffer[i - 1] >= 0)) {
      zcr++;
    }
  }
  zcr = zcr / audioBuffer.length;

  // Estimate F0 (simplified autocorrelation)
  const f0 = estimateF0(audioBuffer, sampleRate);

  // Calculate clarity (simplified - based on energy distribution)
  const clarity = Math.min(1, energy * 2);

  // Calculate fluency (simplified - based on ZCR)
  const fluency = Math.max(0, Math.min(1, 1 - zcr * 2));

  return {
    energy: Math.min(1, energy),
    f0,
    zcr,
    clarity,
    fluency
  };
};

/**
 * Estimate fundamental frequency using autocorrelation
 */
const estimateF0 = (buffer: Float32Array, sampleRate: number): number => {
  const minPeriod = Math.floor(sampleRate / 400); // 400 Hz max
  const maxPeriod = Math.floor(sampleRate / 80);  // 80 Hz min

  let maxCorrelation = 0;
  let bestPeriod = minPeriod;

  for (let period = minPeriod; period < maxPeriod; period++) {
    let correlation = 0;
    for (let i = 0; i < buffer.length - period; i++) {
      correlation += buffer[i] * buffer[i + period];
    }
    correlation /= (buffer.length - period);

    if (correlation > maxCorrelation) {
      maxCorrelation = correlation;
      bestPeriod = period;
    }
  }

  return sampleRate / bestPeriod;
};

