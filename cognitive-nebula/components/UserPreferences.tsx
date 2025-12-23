import React, { useState, useEffect } from 'react';

export interface UserPreferences {
  // Display
  fontSize: 'small' | 'medium' | 'large';
  colorBlindMode: boolean;
  highContrast: boolean;
  reduceMotion: boolean;
  
  // Audio
  audioFeedback: boolean;
  audioVolume: number;
  voiceSensitivity: number;
  
  // Therapeutic
  showHints: boolean;
  difficultyLevel: 'easy' | 'medium' | 'hard';
  practiceMode: 'guided' | 'free' | 'challenge';
  
  // Notifications
  sessionReminders: boolean;
  progressAlerts: boolean;
  achievementNotifications: boolean;
  
  // Privacy
  dataCollection: boolean;
  analytics: boolean;
}

interface UserPreferencesProps {
  preferences: UserPreferences;
  onPreferencesChange: (prefs: Partial<UserPreferences>) => void;
}

const defaultPreferences: UserPreferences = {
  fontSize: 'medium',
  colorBlindMode: false,
  highContrast: false,
  reduceMotion: false,
  audioFeedback: true,
  audioVolume: 0.7,
  voiceSensitivity: 0.5,
  showHints: true,
  difficultyLevel: 'medium',
  practiceMode: 'guided',
  sessionReminders: true,
  progressAlerts: true,
  achievementNotifications: true,
  dataCollection: false,
  analytics: false
};

/**
 * User preferences management component
 * Allows users to customize their experience
 */
const UserPreferences: React.FC<UserPreferencesProps> = ({
  preferences,
  onPreferencesChange
}) => {
  const [localPrefs, setLocalPrefs] = useState<UserPreferences>(preferences);

  useEffect(() => {
    // Apply preferences to document
    document.documentElement.style.fontSize = 
      localPrefs.fontSize === 'small' ? '14px' :
      localPrefs.fontSize === 'large' ? '18px' : '16px';

    if (localPrefs.reduceMotion) {
      document.documentElement.style.setProperty('--animation-duration', '0.01ms');
    }

    if (localPrefs.highContrast) {
      document.documentElement.classList.add('high-contrast');
    } else {
      document.documentElement.classList.remove('high-contrast');
    }
  }, [localPrefs]);

  const handleChange = (key: keyof UserPreferences, value: any) => {
    const newPrefs = { ...localPrefs, [key]: value };
    setLocalPrefs(newPrefs);
    onPreferencesChange({ [key]: value });
  };

  return (
    <div className="space-y-6">
      {/* Display Settings */}
      <section>
        <h3 className="text-cyan-300 font-bold text-lg mb-4">Display</h3>
        <div className="space-y-4">
          <div>
            <label className="text-gray-300 text-sm mb-2 block">Font Size</label>
            <select
              value={localPrefs.fontSize}
              onChange={(e) => handleChange('fontSize', e.target.value)}
              className="w-full bg-gray-800 border border-cyan-500/40 rounded px-3 py-2 text-white"
            >
              <option value="small">Small</option>
              <option value="medium">Medium</option>
              <option value="large">Large</option>
            </select>
          </div>

          <div className="flex items-center justify-between">
            <label className="text-gray-300 text-sm">Color Blind Mode</label>
            <input
              type="checkbox"
              checked={localPrefs.colorBlindMode}
              onChange={(e) => handleChange('colorBlindMode', e.target.checked)}
              className="w-4 h-4"
            />
          </div>

          <div className="flex items-center justify-between">
            <label className="text-gray-300 text-sm">High Contrast</label>
            <input
              type="checkbox"
              checked={localPrefs.highContrast}
              onChange={(e) => handleChange('highContrast', e.target.checked)}
              className="w-4 h-4"
            />
          </div>

          <div className="flex items-center justify-between">
            <label className="text-gray-300 text-sm">Reduce Motion</label>
            <input
              type="checkbox"
              checked={localPrefs.reduceMotion}
              onChange={(e) => handleChange('reduceMotion', e.target.checked)}
              className="w-4 h-4"
            />
          </div>
        </div>
      </section>

      {/* Audio Settings */}
      <section>
        <h3 className="text-cyan-300 font-bold text-lg mb-4">Audio</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <label className="text-gray-300 text-sm">Audio Feedback</label>
            <input
              type="checkbox"
              checked={localPrefs.audioFeedback}
              onChange={(e) => handleChange('audioFeedback', e.target.checked)}
              className="w-4 h-4"
            />
          </div>

          <div>
            <label className="text-gray-300 text-sm mb-2 block">
              Volume: {(localPrefs.audioVolume * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={localPrefs.audioVolume}
              onChange={(e) => handleChange('audioVolume', parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="text-gray-300 text-sm mb-2 block">
              Voice Sensitivity: {(localPrefs.voiceSensitivity * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={localPrefs.voiceSensitivity}
              onChange={(e) => handleChange('voiceSensitivity', parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
      </section>

      {/* Therapeutic Settings */}
      <section>
        <h3 className="text-cyan-300 font-bold text-lg mb-4">Therapeutic</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <label className="text-gray-300 text-sm">Show Hints</label>
            <input
              type="checkbox"
              checked={localPrefs.showHints}
              onChange={(e) => handleChange('showHints', e.target.checked)}
              className="w-4 h-4"
            />
          </div>

          <div>
            <label className="text-gray-300 text-sm mb-2 block">Difficulty Level</label>
            <select
              value={localPrefs.difficultyLevel}
              onChange={(e) => handleChange('difficultyLevel', e.target.value)}
              className="w-full bg-gray-800 border border-cyan-500/40 rounded px-3 py-2 text-white"
            >
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
            </select>
          </div>

          <div>
            <label className="text-gray-300 text-sm mb-2 block">Practice Mode</label>
            <select
              value={localPrefs.practiceMode}
              onChange={(e) => handleChange('practiceMode', e.target.value)}
              className="w-full bg-gray-800 border border-cyan-500/40 rounded px-3 py-2 text-white"
            >
              <option value="guided">Guided</option>
              <option value="free">Free</option>
              <option value="challenge">Challenge</option>
            </select>
          </div>
        </div>
      </section>

      {/* Privacy Settings */}
      <section>
        <h3 className="text-cyan-300 font-bold text-lg mb-4">Privacy</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <label className="text-gray-300 text-sm">Data Collection</label>
            <input
              type="checkbox"
              checked={localPrefs.dataCollection}
              onChange={(e) => handleChange('dataCollection', e.target.checked)}
              className="w-4 h-4"
            />
          </div>

          <div className="flex items-center justify-between">
            <label className="text-gray-300 text-sm">Analytics</label>
            <input
              type="checkbox"
              checked={localPrefs.analytics}
              onChange={(e) => handleChange('analytics', e.target.checked)}
              className="w-4 h-4"
            />
          </div>
        </div>
      </section>
    </div>
  );
};

export { defaultPreferences };
export default UserPreferences;

