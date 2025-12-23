export interface NavItem {
  label: string;
  path: string;
}

export enum DemoPhase {
  IDLE = 'idle',
  LISTENING = 'listening',
  PROCESSING = 'processing',
  PLAYBACK = 'playback'
}

export interface MetricCardProps {
  label: string;
  value: string;
  detail: string;
  trend?: 'up' | 'down' | 'neutral';
}
