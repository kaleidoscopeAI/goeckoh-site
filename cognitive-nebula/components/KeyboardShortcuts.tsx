import React, { useEffect, useState } from 'react';

interface Shortcut {
  key: string;
  description: string;
  action: () => void;
  modifier?: 'ctrl' | 'alt' | 'shift' | 'meta';
}

interface KeyboardShortcutsProps {
  shortcuts: Shortcut[];
  showHelp?: boolean;
  onToggleHelp?: () => void;
}

/**
 * Keyboard shortcuts handler and help display
 * Provides keyboard navigation and quick actions
 */
export const useKeyboardShortcuts = (shortcuts: Shortcut[]) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      shortcuts.forEach(({ key, action, modifier }) => {
        const modifierPressed = 
          (!modifier || modifier === 'ctrl' && e.ctrlKey) ||
          (modifier === 'alt' && e.altKey) ||
          (modifier === 'shift' && e.shiftKey) ||
          (modifier === 'meta' && e.metaKey);

        if (e.key.toLowerCase() === key.toLowerCase() && modifierPressed) {
          e.preventDefault();
          action();
        }
      });
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts]);
};

const KeyboardShortcuts: React.FC<KeyboardShortcutsProps> = ({
  shortcuts,
  showHelp = false,
  onToggleHelp
}) => {
  if (!showHelp) return null;

  return (
    <div className="fixed bottom-4 right-4 bg-black/90 backdrop-blur-lg rounded-xl p-6 border border-cyan-500/30 shadow-2xl max-w-md z-50">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-cyan-300 font-bold text-lg">Keyboard Shortcuts</h3>
        <button
          onClick={onToggleHelp}
          className="text-gray-400 hover:text-white transition-colors"
          aria-label="Close shortcuts help"
        >
          ×
        </button>
      </div>
      <div className="space-y-2">
        {shortcuts.map((shortcut, i) => (
          <div key={i} className="flex items-center justify-between text-sm">
            <span className="text-gray-300">{shortcut.description}</span>
            <kbd className="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-cyan-400 font-mono text-xs">
              {shortcut.modifier && `${shortcut.modifier}+`}{shortcut.key}
            </kbd>
          </div>
        ))}
      </div>
    </div>
  );
};

export default KeyboardShortcuts;

