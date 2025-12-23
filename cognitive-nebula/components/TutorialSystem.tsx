import React, { useState, useEffect } from 'react';
import AnimatedPanel from './AnimatedPanel';

interface TutorialStep {
  id: string;
  title: string;
  content: string;
  target?: string; // CSS selector for element to highlight
  position?: 'top' | 'bottom' | 'left' | 'right' | 'center';
  action?: () => void;
}

interface TutorialSystemProps {
  steps: TutorialStep[];
  onComplete?: () => void;
  onSkip?: () => void;
  showOnFirstVisit?: boolean;
}

/**
 * Interactive tutorial system
 * Guides users through the application with step-by-step instructions
 */
const TutorialSystem: React.FC<TutorialSystemProps> = ({
  steps,
  onComplete,
  onSkip,
  showOnFirstVisit = true
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isVisible, setIsVisible] = useState(false);
  const [hasCompleted, setHasCompleted] = useState(false);

  useEffect(() => {
    if (showOnFirstVisit) {
      const tutorialCompleted = localStorage.getItem('tutorial-completed');
      if (!tutorialCompleted) {
        setIsVisible(true);
      }
    }
  }, [showOnFirstVisit]);

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSkip = () => {
    setIsVisible(false);
    localStorage.setItem('tutorial-completed', 'skipped');
    onSkip?.();
  };

  const handleComplete = () => {
    setIsVisible(false);
    setHasCompleted(true);
    localStorage.setItem('tutorial-completed', 'true');
    onComplete?.();
  };

  if (!isVisible || hasCompleted || steps.length === 0) {
    return null;
  }

  const step = steps[currentStep];
  const progress = ((currentStep + 1) / steps.length) * 100;

  return (
    <>
      {/* Overlay */}
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40" />

      {/* Tutorial Panel */}
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        <div className="bg-black/90 backdrop-blur-lg rounded-xl p-6 max-w-md w-full border border-cyan-500/30 shadow-2xl">
          {/* Progress Bar */}
          <div className="mb-4">
            <div className="flex justify-between text-xs text-gray-400 mb-2">
              <span>Step {currentStep + 1} of {steps.length}</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div
                className="bg-cyan-400 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          {/* Content */}
          <div className="mb-6">
            <h3 className="text-cyan-300 font-bold text-xl mb-2">{step.title}</h3>
            <p className="text-gray-300">{step.content}</p>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between">
            <button
              onClick={handleSkip}
              className="text-gray-400 hover:text-white transition-colors text-sm"
            >
              Skip Tutorial
            </button>

            <div className="flex gap-2">
              {currentStep > 0 && (
                <button
                  onClick={handlePrevious}
                  className="bg-gray-700 hover:bg-gray-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
                >
                  Previous
                </button>
              )}
              <button
                onClick={handleNext}
                className="bg-cyan-600 hover:bg-cyan-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
              >
                {currentStep === steps.length - 1 ? 'Complete' : 'Next'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

/**
 * Hook to trigger tutorial
 */
export const useTutorial = () => {
  const [showTutorial, setShowTutorial] = useState(false);

  const startTutorial = () => {
    localStorage.removeItem('tutorial-completed');
    setShowTutorial(true);
  };

  return { showTutorial, startTutorial, setShowTutorial };
};

export default TutorialSystem;

