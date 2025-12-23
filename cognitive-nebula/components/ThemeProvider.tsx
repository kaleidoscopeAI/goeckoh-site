import React, { createContext, useContext, useState, useEffect } from 'react';

type Theme = 'dark' | 'light' | 'auto';
type ColorScheme = 'cyan' | 'purple' | 'green' | 'orange';

interface ThemeContextType {
  theme: Theme;
  colorScheme: ColorScheme;
  setTheme: (theme: Theme) => void;
  setColorScheme: (scheme: ColorScheme) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
};

interface ThemeProviderProps {
  children: React.ReactNode;
  defaultTheme?: Theme;
  defaultColorScheme?: ColorScheme;
}

/**
 * Theme provider for application-wide theming
 * Supports dark/light/auto themes and color schemes
 */
export const ThemeProvider: React.FC<ThemeProviderProps> = ({
  children,
  defaultTheme = 'dark',
  defaultColorScheme = 'cyan'
}) => {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem('theme');
    return (saved as Theme) || defaultTheme;
  });

  const [colorScheme, setColorScheme] = useState<ColorScheme>(() => {
    const saved = localStorage.getItem('colorScheme');
    return (saved as ColorScheme) || defaultColorScheme;
  });

  useEffect(() => {
    localStorage.setItem('theme', theme);
    localStorage.setItem('colorScheme', colorScheme);

    const root = document.documentElement;
    
    // Apply theme
    if (theme === 'auto') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.classList.toggle('dark', prefersDark);
      root.classList.toggle('light', !prefersDark);
    } else {
      root.classList.toggle('dark', theme === 'dark');
      root.classList.toggle('light', theme === 'light');
    }

    // Apply color scheme
    root.setAttribute('data-color-scheme', colorScheme);
  }, [theme, colorScheme]);

  return (
    <ThemeContext.Provider value={{ theme, colorScheme, setTheme, setColorScheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

/**
 * Theme toggle component
 */
export const ThemeToggle: React.FC = () => {
  const { theme, setTheme, colorScheme, setColorScheme } = useTheme();

  return (
    <div className="flex items-center gap-4">
      <select
        value={theme}
        onChange={(e) => setTheme(e.target.value as Theme)}
        className="bg-gray-800 border border-cyan-500/40 rounded px-2 py-1 text-sm text-white"
      >
        <option value="dark">Dark</option>
        <option value="light">Light</option>
        <option value="auto">Auto</option>
      </select>
      <select
        value={colorScheme}
        onChange={(e) => setColorScheme(e.target.value as ColorScheme)}
        className="bg-gray-800 border border-cyan-500/40 rounded px-2 py-1 text-sm text-white"
      >
        <option value="cyan">Cyan</option>
        <option value="purple">Purple</option>
        <option value="green">Green</option>
        <option value="orange">Orange</option>
      </select>
    </div>
  );
};

