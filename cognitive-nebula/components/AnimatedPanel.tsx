import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

interface AnimatedPanelProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  position?: 'left' | 'right' | 'top' | 'bottom' | 'center';
  size?: 'small' | 'medium' | 'large' | 'full';
  title?: string;
  className?: string;
}

/**
 * Animated panel component with smooth transitions
 * Supports slide-in animations from different directions
 */
const AnimatedPanel: React.FC<AnimatedPanelProps> = ({
  isOpen,
  onClose,
  children,
  position = 'right',
  size = 'medium',
  title,
  className = ''
}) => {
  const [isAnimating, setIsAnimating] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);
  const backdropRef = useRef<HTMLDivElement>(null);

  const sizeClasses = {
    small: 'max-w-sm',
    medium: 'max-w-md',
    large: 'max-w-2xl',
    full: 'max-w-full'
  };

  const positionClasses = {
    left: 'left-0 top-0 h-full',
    right: 'right-0 top-0 h-full',
    top: 'top-0 left-0 w-full',
    bottom: 'bottom-0 left-0 w-full',
    center: 'top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2'
  };

  const transformClasses = {
    left: isOpen ? 'translate-x-0' : '-translate-x-full',
    right: isOpen ? 'translate-x-0' : 'translate-x-full',
    top: isOpen ? 'translate-y-0' : '-translate-y-full',
    bottom: isOpen ? 'translate-y-0' : 'translate-y-full',
    center: isOpen ? 'scale-100 opacity-100' : 'scale-95 opacity-0'
  };

  useEffect(() => {
    if (isOpen) {
      setIsAnimating(true);
      // Prevent body scroll when panel is open
      document.body.style.overflow = 'hidden';
    } else {
      // Allow body scroll when panel is closed
      document.body.style.overflow = '';
    }

    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  if (!isOpen && !isAnimating) return null;

  const panelContent = (
    <>
      {/* Backdrop */}
      <div
        ref={backdropRef}
        className={`fixed inset-0 bg-black/70 backdrop-blur-sm z-40 transition-opacity duration-300 ${
          isOpen ? 'opacity-100' : 'opacity-0'
        }`}
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <div
        ref={panelRef}
        className={`fixed z-50 bg-black/90 backdrop-blur-lg border border-cyan-500/30 shadow-2xl ${sizeClasses[size]} ${positionClasses[position]} ${transformClasses[position]} transition-all duration-300 ease-out ${className}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? 'panel-title' : undefined}
      >
        {title && (
          <div className="flex items-center justify-between p-4 border-b border-cyan-500/30">
            <h2 id="panel-title" className="text-cyan-300 font-bold text-lg">
              {title}
            </h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white transition-colors p-1 rounded hover:bg-gray-700"
              aria-label="Close panel"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
        <div className="overflow-y-auto h-full">
          {children}
        </div>
      </div>
    </>
  );

  return createPortal(panelContent, document.body);
};

export default AnimatedPanel;

