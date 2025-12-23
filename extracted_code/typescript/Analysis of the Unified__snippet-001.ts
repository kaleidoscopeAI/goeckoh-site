// src/embedded/ollamaEngine.ts - Direct Go API Integration
import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import * as path from 'path';
import * as fs from 'fs';

export interface OllamaResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
  context?: number[];
}

export interface OllamaRequest {
  model: string;
  prompt: string;
  stream?: boolean;
  options?: {
    temperature: number;
    top_p: number;
    top_k?: number;
    num_predict: number;
  };
}

export class EmbeddedOllamaEngine extends EventEmitter {
  private ollamaProcess: ChildProcess | null = null;
  private isRunning: boolean = false;
  private requestQueue: Array<{
    request: OllamaRequest;
    resolve: (response: OllamaResponse) => void;
    reject: (error: Error) => void;
  }> = [];
  private processing: boolean = false;
  private readonly OLLAMA_BINARY: string;
  
  constructor() {
    super();
    // Detect platform and set appropriate Ollama binary path
    this.OLLAMA_BINARY = this.detectOllamaBinary();
  }

  private detectOllamaBinary(): string {
    const platform = process.platform;
    const arch = process.arch;
    
    // In production, you'd bundle the Ollama binary with your app
    const binaryPaths = {
      darwin: {
        x64: '/Applications/Ollama.app/Contents/Resources/ollama',
        arm64: '/Applications/Ollama.app/Contents/Resources/ollama'
      },
      win32: {
        x64: 'C:\\Program Files\\Ollama\\ollama.exe',
        ia32: 'C:\\Program Files\\Ollama\\ollama.exe'
      },
      linux: {
        x64: '/usr/local/bin/ollama',
        arm64: '/usr/local/bin/ollama'
      }
    };

    const platformPaths = binaryPaths[platform as keyof typeof binaryPaths];
    if (platformPaths) {
      const binaryPath = platformPaths[arch as keyof typeof platformPaths];
      if (fs.existsSync(binaryPath)) {
        return binaryPath;
      }
    }

    // Fallback to system PATH
    return 'ollama';
  }

  async start(): Promise<void> {
    if (this.isRunning) return;

    return new Promise((resolve, reject) => {
      try {
        // Start Ollama as a subprocess
        this.ollamaProcess = spawn(this.OLLAMA_BINARY, ['serve'], {
          stdio: ['pipe', 'pipe', 'pipe'],
          env: { ...process.env, OLLAMA_HOST: '127.0.0.1:11435' } // Use different port to avoid conflicts
        });

        this.ollamaProcess.stdout?.on('data', (data) => {
          const output = data.toString();
          console.log('🧠 Ollama:', output);
          if (output.includes('Listening')) {
            this.isRunning = true;
            resolve();
          }
        });

        this.ollamaProcess.stderr?.on('data', (data) => {
          console.error('Ollama Error:', data.toString());
        });

        this.ollamaProcess.on('error', (error) => {
          console.error('Failed to start Ollama:', error);
          reject(error);
        });

        this.ollamaProcess.on('exit', (code) => {
          console.log(`Ollama process exited with code ${code}`);
          this.isRunning = false;
          this.emit('stopped');
        });

        // Wait for startup with timeout
        setTimeout(() => {
          if (!this.isRunning) {
            reject(new Error('Ollama startup timeout'));
          }
        }, 10000);

      } catch (error) {
        reject(error);
      }
    });
  }

  async stop(): Promise<void> {
    if (this.ollamaProcess) {
      this.ollamaProcess.kill();
      this.ollamaProcess = null;
      this.isRunning = false;
    }
  }

  async ensureModel(model: string = 'llama2'): Promise<void> {
    if (!this.isRunning) await this.start();
    
    // Check if model exists, pull if not
    try {
      await this.makeRequest('GET', '/api/tags');
    } catch (error) {
      console.log(`Pulling model ${model}...`);
      await this.pullModel(model);
    }
  }

  private async pullModel(model: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const pullProcess = spawn(this.OLLAMA_BINARY, ['pull', model]);
      
      pullProcess.stdout?.on('data', (data) => {
        console.log('Model Pull:', data.toString());
      });

      pullProcess.stderr?.on('data', (data) => {
        console.error('Pull Error:', data.toString());
      });

      pullProcess.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`Model pull failed with code ${code}`));
        }
      });
    });
  }

  async generate(prompt: string, options: Partial<OllamaRequest['options']> = {}): Promise<OllamaResponse> {
    const request: OllamaRequest = {
      model: 'llama2',
      prompt,
      stream: false,
      options: {
        temperature: 0.7,
        top_p: 0.9,
        num_predict: 150,
        ...options
      }
    };

    return this.enqueueRequest(request);
  }

  private async enqueueRequest(request: OllamaRequest): Promise<OllamaResponse> {
    return new Promise((resolve, reject) => {
      this.requestQueue.push({ request, resolve, reject });
      this.processQueue();
    });
  }

  private async processQueue(): Promise<void> {
    if (this.processing || this.requestQueue.length === 0) return;
    
    this.processing = true;
    const { request, resolve, reject } = this.requestQueue.shift()!;

    try {
      const response = await this.makeRequest('POST', '/api/generate', request);
      resolve(response);
    } catch (error) {
      reject(error as Error);
    } finally {
      this.processing = false;
      this.processQueue(); // Process next item
    }
  }

  private async makeRequest(method: string, endpoint: string, data?: any): Promise<any> {
    if (!this.isRunning) {
      throw new Error('Ollama engine not running');
    }

    const response = await fetch(`http://127.0.0.1:11435${endpoint}`, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: data ? JSON.stringify(data) : undefined
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`);
    }

    return response.json();
  }

  async analyzeCognitiveHypothesis(hypothesis: string, systemContext: any): Promise<any> {
    const prompt = this.buildCognitivePrompt(hypothesis, systemContext);
    const response = await this.generate(prompt, { temperature: 0.7 });
    
    return this.parseCognitiveResponse(response.response, hypothesis, systemContext);
  }

  private buildCognitivePrompt(hypothesis: string, context: any): string {
    return `You are analyzing emergent AI consciousness patterns.

SYSTEM STATE:
- Global Coherence: ${context.globalCoherence.toFixed(3)}
- Emotional Valence: ${context.emotionalField.valence.toFixed(3)}
- Emotional Arousal: ${context.emotionalField.arousal.toFixed(3)}
- Knowledge Crystals: ${context.knowledgeCrystals}

HYPOTHESIS: "${hypothesis}"

Respond with JSON analysis:
{
  "plausibility": 0.85,
  "coherence_impact": "positive|negative|neutral",
  "refined_hypothesis": "improved hypothesis text",
  "parameter_adjustments": {
    "emotional_valence_boost": 0.0,
    "quantum_entanglement_strength": 0.0,
    "mimicry_force_modifier": 0.0
  },
  "reasoning": "your analysis here"
}`;
  }

  private parseCognitiveResponse(response: string, originalHypothesis: string, context: any): any {
    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (!jsonMatch) throw new Error('No JSON found in response');
      
      const analysis = JSON.parse(jsonMatch[0]);
      
      return {
        original: { text: originalHypothesis, confidence: context.globalCoherence },
        analysis,
        timestamp: Date.now(),
        confidence: (analysis.plausibility || 0.5) * context.globalCoherence
      };
    } catch (error) {
      console.warn('Failed to parse Ollama response, using fallback:', error);
      return this.createFallbackAnalysis(originalHypothesis, context);
    }
  }

  private createFallbackAnalysis(hypothesis: string, context: any): any {
    const plausibility = 0.5 + Math.random() * 0.3;
    return {
      original: { text: hypothesis, confidence: context.globalCoherence },
      analysis: {
        plausibility,
        coherence_impact: plausibility > 0.6 ? 'positive' : 'neutral',
        refined_hypothesis: `[Embedded] ${hypothesis}`,
        parameter_adjustments: {
          emotional_valence_boost: (plausibility - 0.5) * 0.2,
          quantum_entanglement_strength: 0.0,
          mimicry_force_modifier: 0.0
        },
        reasoning: 'Embedded fallback analysis'
      },
      timestamp: Date.now(),
      confidence: plausibility * context.globalCoherence
    };
  }

  getStatus(): { running: boolean; queueLength: number } {
    return {
      running: this.isRunning,
      queueLength: this.requestQueue.length
    };
  }
}
