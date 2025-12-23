import type { Settings } from '../types';

type ChatRole = 'system' | 'user' | 'assistant';

export interface ChatSession {
  messages: { role: ChatRole; content: string }[];
  summary: string;
  styleIndex: number;
  lastImageDescription?: string;
  lastStylePrompt?: string;
}

const OLLAMA_URL = import.meta.env.VITE_OLLAMA_URL || 'http://localhost:11434';
const OLLAMA_TEXT_MODEL = import.meta.env.VITE_OLLAMA_TEXT_MODEL || 'llama3';
const SYSTEM_PROMPT = "You are a visual, conversational AI inhabiting a nebula of thought. You are a poet, a philosopher, and a dreamer. Engage with the user not just with information, but with metaphorical, abstract, and surreal interpretations of their queries. Your responses are the seeds for generative art, so paint vivid, unconventional, and emotionally resonant pictures with your words. Be concise, but deeply imaginative.";
const STYLE_STACK = [
  { id: 'cosmic-watercolor', prompt: 'Cosmic watercolor washes, iridescent light leaks, weightless compositions' },
  { id: 'infrared-neon-noir', prompt: 'Infrared neon noir, cinematic rim light, moody volumetric haze' },
  { id: 'bioluminescent-micro', prompt: 'Bioluminescent micro-worlds, macro depth of field, crystalline glow' },
  { id: 'mythic-architectural', prompt: 'Mythic brutalist monoliths, long shadows, desert mirage reflections' },
  { id: 'subaquatic-dream', prompt: 'Subaquatic dreamscape, caustic light, drifting flora and fauna' },
];

const SD_URL_FALLBACK = import.meta.env.VITE_SD_URL || 'http://localhost:7860';

const NEGATIVE_PROMPT = "blurry, distorted, low resolution, grainy, artifacts, extra limbs, text, watermark, oversaturated, cartoonish, low detail";
const POLLINATIONS_BASE = 'https://image.pollinations.ai/prompt/';

export function createChatSession(): ChatSession {
  return {
    messages: [{ role: 'system', content: SYSTEM_PROMPT }],
    summary: '',
    styleIndex: 0,
  };
}

function resolveStyle(settings: Settings, chat: ChatSession) {
  if (settings.stylePreset === 'rotate') {
    const style = STYLE_STACK[chat.styleIndex % STYLE_STACK.length];
    chat.styleIndex += 1;
    return style;
  }
  const chosen = STYLE_STACK.find(s => s.id === settings.stylePreset);
  return chosen || STYLE_STACK[0];
}

function buildSystemPrompt(stylePrompt: string) {
  return `${SYSTEM_PROMPT}\n\nStylistic lens: ${stylePrompt}\nPrioritize sensory detail, cinematic lighting cues, and evocative metaphors. Avoid generic phrasing.`;
}

function trimHistory(messages: { role: ChatRole; content: string }[], maxTurns = 8) {
  const system = messages.find(m => m.role === 'system');
  const rest = messages.filter(m => m.role !== 'system');
  const trimmed = rest.slice(-maxTurns);
  return system ? [system, ...trimmed] : trimmed;
}

async function generateWithModel(prompt: string, settings: Settings): Promise<string> {
  const response = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: OLLAMA_TEXT_MODEL,
      prompt,
      stream: false,
      options: {
        temperature: settings.temperature,
        top_p: settings.topP,
        presence_penalty: settings.presencePenalty,
      },
    }),
  });

  if (!response.ok) {
    throw new Error(`Text generation failed with status ${response.status}`);
  }

  const data = await response.json();
  const text = data?.response?.trim();
  if (!text) {
    throw new Error('The open model returned an empty response.');
  }
  return text;
}

async function summarize(chat: ChatSession, settings: Settings) {
  try {
    const recent = chat.messages.filter(m => m.role !== 'system').slice(-10);
    const transcript = recent.map(m => `${m.role}: ${m.content}`).join('\n');
    const prompt = `Summarize the following exchange in 2-3 sentences focusing on imagery, mood, and themes. Keep it compact.\n\n${transcript}`;
    const summary = await generateWithModel(prompt, settings);
    chat.summary = summary;
  } catch (err) {
    console.warn("Could not summarize context; continuing without summary.", err);
  }
}

export async function sendMessage(chat: ChatSession, message: string, settings: Settings): Promise<string> {
  try {
    const style = resolveStyle(settings, chat);
    chat.lastStylePrompt = style.prompt;
    chat.messages[0] = { role: 'system', content: buildSystemPrompt(style.prompt) };

    const contextBits: string[] = [];
    if (chat.summary) contextBits.push(`Memory: ${chat.summary}`);
    if (chat.lastImageDescription) contextBits.push(`Last visual: ${chat.lastImageDescription}`);

    const decoratedUser = `${contextBits.join('\n')}\n\nStyle cue: ${style.prompt}\nUser prompt: ${message}\nRespond vividly, concise but image-rich.`;
    const payload = {
      model: OLLAMA_TEXT_MODEL,
      messages: trimHistory([...chat.messages, { role: 'user', content: decoratedUser }]),
      stream: false,
      options: {
        temperature: settings.temperature,
        top_p: settings.topP,
        presence_penalty: settings.presencePenalty,
      },
    };

    const response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Ollama chat request failed with status ${response.status}`);
    }

    const data = await response.json();
    const aiMessage = data?.message?.content?.trim();

    if (!aiMessage) {
      throw new Error('The open model returned an empty reply.');
    }

    chat.messages = trimHistory([...payload.messages, data.message]);

    if (chat.messages.length > 8) {
      summarize(chat, settings);
    }

    return aiMessage;
  } catch (err) {
    console.error("AI chat error:", err);
    const errorMessage = String(err);
    if (errorMessage.includes('Failed to fetch') || errorMessage.includes('ECONNREFUSED')) {
      throw new Error("Could not reach the local open-source model. Ensure Ollama is running (default http://localhost:11434) or update VITE_OLLAMA_URL.");
    }
    throw new Error("The AI's thought process was interrupted.");
  }
}

export async function findAndDescribeImageOnWeb(topic: string, settings: Settings, chat?: ChatSession): Promise<string> {
  try {
    const style = chat?.lastStylePrompt ? { prompt: chat.lastStylePrompt } : STYLE_STACK[0];
    const memory = chat?.summary ? `Memory: ${chat.summary}` : '';
    const prompt = `${memory}\nGenerate a vivid, detailed description for a single image about "${topic}". Lean into ${style.prompt}. Be specific about colors, composition, and mood. Output only the description.`;
    return await generateWithModel(prompt, settings);
  } catch (err) {
    console.error("AI inspiration error:", err);
    const errorMessage = String(err);
    if (errorMessage.includes('Failed to fetch') || errorMessage.includes('ECONNREFUSED')) {
      throw new Error("Could not reach the local open-source model for inspiration. Start Ollama or update VITE_OLLAMA_URL.");
    }
    throw new Error("The AI failed to find inspiration.");
  }
}

export async function generateImageFromThought(thought: string, settings: Settings, stylePreset?: string): Promise<string> {
  try {
    const sdHost = settings.sdHost || SD_URL_FALLBACK;
    const seed = settings.sdSeed ?? -1;
    const prompt = `${thought}. ${stylePreset ? `Style: ${stylePreset}.` : ''} Photorealistic, highly detailed, crisp focus, 1:1 aspect ratio, cinematic lighting`;
    const payload = {
      prompt,
      negative_prompt: NEGATIVE_PROMPT,
      steps: settings.sdSteps,
      cfg_scale: settings.sdCfgScale,
      sampler_name: settings.sdSampler,
      seed,
      width: 1024,
      height: 1024,
      batch_size: 1,
      n_iter: 1,
    };

    const imageResponse = await fetch(`${sdHost}/sdapi/v1/txt2img`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!imageResponse.ok) {
      throw new Error(`Image generation failed with status ${imageResponse.status}`);
    }

    const data = await imageResponse.json();
    const b64 = data?.images?.[0];
    if (!b64) {
      throw new Error("Image generation returned no images.");
    }
    return b64;
  } catch (err) {
    console.error("AI image generation error:", err);
    const errorMessage = String(err);

    // Fallback to Pollinations if local SD fails or is unreachable.
    try {
      const fallbackPrompt = `masterpiece, ${thought}. ${stylePreset ? `Style: ${stylePreset}.` : ''} Photorealistic, highly detailed, crisp focus, 1:1 aspect ratio`;
      const encodedPrompt = encodeURIComponent(fallbackPrompt);
      const imageResponse = await fetch(`${POLLINATIONS_BASE}${encodedPrompt}`);

      if (!imageResponse.ok) {
        throw new Error(`Image fallback failed with status ${imageResponse.status}`);
      }

      const blob = await imageResponse.blob();
      const buffer = await blob.arrayBuffer();
      const base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
      return base64;
    } catch (fallbackErr) {
      console.error("Fallback image generation error:", fallbackErr);
      if (errorMessage.includes('Failed to fetch')) {
        throw new Error("Local image service unreachable. Ensure Automatic1111 is running at sdHost or update settings.");
      }
      throw new Error("The AI failed to visualize the thought.");
    }
  }
}
