import React, { useState } from 'react';

export type ViewMode = 'split' | 'focus' | 'fullscreen' | 'grid' | 'therapeutic';
export type PanelLayout = 'default' | 'minimal' | 'expanded';

interface MultiViewLayoutProps {
  children: React.ReactNode;
  defaultMode?: ViewMode;
  onModeChange?: (mode: ViewMode) => void;
}

interface ViewModeContextType {
  currentMode: ViewMode;
  setMode: (mode: ViewMode) => void;
  panelLayout: PanelLayout;
  setPanelLayout: (layout: PanelLayout) => void;
}

const ViewModeContext = React.createContext<ViewModeContextType | undefined>(undefined);

export const useViewMode = () => {
  const context = React.useContext(ViewModeContext);
  if (!context) {
    throw new Error('useViewMode must be used within MultiViewLayout');
  }
  return context;
};

/**
 * Multi-view layout system
 * Provides different viewing modes for different use cases
 */
const MultiViewLayout: React.FC<MultiViewLayoutProps> = ({
  children,
  defaultMode = 'split',
  onModeChange
}) => {
  const [currentMode, setCurrentMode] = useState<ViewMode>(defaultMode);
  const [panelLayout, setPanelLayout] = useState<PanelLayout>('default');

  const handleModeChange = (mode: ViewMode) => {
    setCurrentMode(mode);
    onModeChange?.(mode);
  };

  const modeClasses = {
    split: 'grid grid-cols-2 gap-4',
    focus: 'flex flex-col',
    fullscreen: 'absolute inset-0',
    grid: 'grid grid-cols-3 gap-4',
    therapeutic: 'flex flex-col space-y-4'
  };

  return (
    <ViewModeContext.Provider value={{ currentMode, setMode: handleModeChange, panelLayout, setPanelLayout }}>
      <div className={`w-full h-full ${modeClasses[currentMode]}`}>
        {children}
      </div>
    </ViewModeContext.Provider>
  );
};

/**
 * View mode selector component
 */
export const ViewModeSelector: React.FC = () => {
  const { currentMode, setMode } = useViewMode();

  const modes: { mode: ViewMode; label: string; icon: string }[] = [
    { mode: 'split', label: 'Split', icon: '⛶' },
    { mode: 'focus', label: 'Focus', icon: '🎯' },
    { mode: 'fullscreen', label: 'Fullscreen', icon: '⛶' },
    { mode: 'grid', label: 'Grid', icon: '⊞' },
    { mode: 'therapeutic', label: 'Therapeutic', icon: '🎤' }
  ];

  return (
    <div className="flex items-center gap-2 bg-black/70 backdrop-blur-lg rounded-lg p-2 border border-cyan-500/30">
      {modes.map(({ mode, label, icon }) => (
        <button
          key={mode}
          onClick={() => setMode(mode)}
          className={`px-3 py-1 rounded transition-colors ${
            currentMode === mode
              ? 'bg-cyan-600 text-white'
              : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
          }`}
          title={label}
        >
          <span className="mr-1">{icon}</span>
          <span className="text-xs">{label}</span>
        </button>
      ))}
    </div>
  );
};

export default MultiViewLayout;

