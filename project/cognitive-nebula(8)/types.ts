export interface Metrics {
  curiosity: number;
  confusion: number;
  coherence: number;
  arousal: number;
  valence: number;
  dominance: number;
  certainty: number;
  resonance: number;
}

export interface ChatMessage {
  role: 'user' | 'model' | 'dream';
  content: string;
}

export interface Settings {
  particleCount: number; // 0 to 1
  colorSaturation: number; // 0 to 1
  movementSpeed: number; // 0 to 1
  showStars: boolean;
  showTrails: boolean;
  temperature: number; // 0 to 2
  topP: number; // 0 to 1
  presencePenalty: number; // 0 to 1+
  stylePreset: string;
  sdHost: string;
  sdSteps: number;
  sdCfgScale: number;
  sdSampler: string;
  sdSeed: number | null; // null = random
}
