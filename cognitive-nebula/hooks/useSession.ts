import { useState, useEffect, useCallback } from 'react';

interface SessionData {
  id: string;
  startTime: number;
  endTime?: number;
  duration: number;
  exercisesCompleted: number;
  averageClarity: number;
  averageFluency: number;
  improvements: string[];
}

interface UseSessionOptions {
  autoStart?: boolean;
  onSessionEnd?: (session: SessionData) => void;
}

/**
 * Custom hook for managing therapy sessions
 * Tracks session data, exercises, and progress
 */
export const useSession = (options: UseSessionOptions = {}) => {
  const { autoStart = false, onSessionEnd } = options;

  const [currentSession, setCurrentSession] = useState<SessionData | null>(null);
  const [isActive, setIsActive] = useState(autoStart);
  const [sessionHistory, setSessionHistory] = useState<SessionData[]>([]);

  // Load session history from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('session-history');
    if (saved) {
      try {
        setSessionHistory(JSON.parse(saved));
      } catch (e) {
        console.error('Failed to load session history', e);
      }
    }
  }, []);

  // Auto-start session if enabled
  useEffect(() => {
    if (autoStart && !currentSession) {
      startSession();
    }
  }, [autoStart]);

  const startSession = useCallback(() => {
    const session: SessionData = {
      id: `session-${Date.now()}`,
      startTime: Date.now(),
      duration: 0,
      exercisesCompleted: 0,
      averageClarity: 0,
      averageFluency: 0,
      improvements: []
    };

    setCurrentSession(session);
    setIsActive(true);
  }, []);

  const endSession = useCallback(() => {
    if (!currentSession) return;

    const endedSession: SessionData = {
      ...currentSession,
      endTime: Date.now(),
      duration: Date.now() - currentSession.startTime
    };

    setCurrentSession(null);
    setIsActive(false);

    // Save to history
    const updatedHistory = [endedSession, ...sessionHistory].slice(0, 100); // Keep last 100
    setSessionHistory(updatedHistory);
    localStorage.setItem('session-history', JSON.stringify(updatedHistory));

    onSessionEnd?.(endedSession);
  }, [currentSession, sessionHistory, onSessionEnd]);

  const completeExercise = useCallback(() => {
    if (!currentSession) return;

    setCurrentSession(prev => prev ? {
      ...prev,
      exercisesCompleted: prev.exercisesCompleted + 1
    } : null);
  }, [currentSession]);

  const updateMetrics = useCallback((clarity: number, fluency: number) => {
    if (!currentSession) return;

    setCurrentSession(prev => prev ? {
      ...prev,
      averageClarity: (prev.averageClarity * prev.exercisesCompleted + clarity) / (prev.exercisesCompleted + 1),
      averageFluency: (prev.averageFluency * prev.exercisesCompleted + fluency) / (prev.exercisesCompleted + 1)
    } : null);
  }, [currentSession]);

  const getSessionTime = useCallback(() => {
    if (!currentSession) return 0;
    return Math.floor((Date.now() - currentSession.startTime) / 1000);
  }, [currentSession]);

  return {
    currentSession,
    isActive,
    sessionHistory,
    startSession,
    endSession,
    completeExercise,
    updateMetrics,
    getSessionTime
  };
};

