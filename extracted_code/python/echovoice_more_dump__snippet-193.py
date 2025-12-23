import fetch from 'node-fetch';
import { NodeState } from './types';

export class LLMAdapter {
  url: string;
  model: string;
  constructor() {
    this.url = process.env.OLLAMA_URL || 'http://localhost:11434';
    this.model = process.env.OLLAMA_MODEL || 'llama2';
  }

  async suggest(batch: Record<string, Partial<NodeState>>): Promise<Record<string, number[]>> {
    const prompt = `Generate semantic deltas for nodes: ${JSON.stringify(batch)}. Keep deltas small and relevant.`;
    const response = await fetch(`${this.url}/api/generate`, {
      method: 'POST',
      body: JSON.stringify({ model: this.model, prompt })
    });
    const data = await response.json();
    const suggestions: Record<string, number[]> = JSON.parse(data.response); // Assume parsed deltas
    return suggestions;
  }
