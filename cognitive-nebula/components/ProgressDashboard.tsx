import React, { useState } from 'react';

interface SessionData {
  date: string;
  duration: number;
  exercisesCompleted: number;
  averageClarity: number;
  averageFluency: number;
  improvements: string[];
}

interface ProgressDashboardProps {
  sessions: SessionData[];
  currentStreak: number;
  totalTime: number;
  onExport?: () => void;
}

/**
 * Progress tracking dashboard
 * Shows therapeutic progress over time with charts and statistics
 */
const ProgressDashboard: React.FC<ProgressDashboardProps> = ({
  sessions,
  currentStreak,
  totalTime,
  onExport
}) => {
  const [selectedPeriod, setSelectedPeriod] = useState<'week' | 'month' | 'all'>('week');

  const filteredSessions = sessions.filter(session => {
    const sessionDate = new Date(session.date);
    const now = new Date();
    if (selectedPeriod === 'week') {
      const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      return sessionDate >= weekAgo;
    } else if (selectedPeriod === 'month') {
      const monthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
      return sessionDate >= monthAgo;
    }
    return true;
  });

  const averageClarity = filteredSessions.length > 0
    ? filteredSessions.reduce((sum, s) => sum + s.averageClarity, 0) / filteredSessions.length
    : 0;

  const averageFluency = filteredSessions.length > 0
    ? filteredSessions.reduce((sum, s) => sum + s.averageFluency, 0) / filteredSessions.length
    : 0;

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <div className="absolute top-4 right-4 bg-black/70 backdrop-blur-lg rounded-xl p-6 border border-cyan-500/30 shadow-2xl max-w-md w-full max-h-[80vh] overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-cyan-300 font-bold text-lg">Progress Dashboard</h3>
        <div className="flex gap-2">
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value as 'week' | 'month' | 'all')}
            className="bg-gray-800 border border-cyan-500/40 rounded px-2 py-1 text-xs text-white"
          >
            <option value="week">Week</option>
            <option value="month">Month</option>
            <option value="all">All Time</option>
          </select>
          {onExport && (
            <button
              onClick={onExport}
              className="bg-cyan-600 hover:bg-cyan-700 text-white text-xs px-3 py-1 rounded transition-colors"
            >
              Export
            </button>
          )}
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-gray-900/50 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Current Streak</div>
          <div className="text-2xl font-bold text-cyan-300">{currentStreak} days</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Total Time</div>
          <div className="text-2xl font-bold text-cyan-300">{formatTime(totalTime)}</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Avg Clarity</div>
          <div className="text-xl font-bold text-green-400">{(averageClarity * 100).toFixed(0)}%</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Avg Fluency</div>
          <div className="text-xl font-bold text-green-400">{(averageFluency * 100).toFixed(0)}%</div>
        </div>
      </div>

      {/* Progress Chart */}
      <div className="mb-4">
        <h4 className="text-sm font-semibold text-cyan-400 mb-2">Clarity Trend</h4>
        <div className="bg-gray-900/50 rounded-lg p-3 h-32 flex items-end gap-1">
          {filteredSessions.slice(-7).map((session, i) => (
            <div key={i} className="flex-1 flex flex-col items-center">
              <div
                className="w-full bg-gradient-to-t from-cyan-600 to-cyan-400 rounded-t transition-all duration-300"
                style={{ height: `${session.averageClarity * 100}%` }}
              />
              <div className="text-xs text-gray-500 mt-1">
                {new Date(session.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Sessions */}
      <div>
        <h4 className="text-sm font-semibold text-cyan-400 mb-2">Recent Sessions</h4>
        <div className="space-y-2">
          {filteredSessions.slice(-5).reverse().map((session, i) => (
            <div key={i} className="bg-gray-900/50 rounded-lg p-3 text-xs">
              <div className="flex justify-between mb-1">
                <span className="text-gray-300">{new Date(session.date).toLocaleDateString()}</span>
                <span className="text-cyan-400">{formatTime(session.duration)}</span>
              </div>
              <div className="flex gap-3 text-gray-400">
                <span>Clarity: {(session.averageClarity * 100).toFixed(0)}%</span>
                <span>Fluency: {(session.averageFluency * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ProgressDashboard;

