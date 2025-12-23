interface VoiceData {
  bubbleState?: {
    radius: number;
    color_r: number;
    color_g: number;
    color_b: number;
    rough: number;
    metal: number;
    spike: number;
    energy?: number;
    f0?: number;
  };
  waveform?: Float32Array;
  spectrum?: Float32Array;
}
