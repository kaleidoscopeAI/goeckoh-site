import React, { useEffect, useState } from 'react';

interface MobileOptimizedProps {
  children: React.ReactNode;
  breakpoint?: number;
}

/**
 * Mobile optimization wrapper
 * Adapts layout and behavior for mobile devices
 */
const MobileOptimized: React.FC<MobileOptimizedProps> = ({
  children,
  breakpoint = 768
}) => {
  const [isMobile, setIsMobile] = useState(false);
  const [touchStart, setTouchStart] = useState<number | null>(null);
  const [touchEnd, setTouchEnd] = useState<number | null>(null);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < breakpoint);
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, [breakpoint]);

  // Swipe detection
  const minSwipeDistance = 50;

  const onTouchStart = (e: React.TouchEvent) => {
    setTouchEnd(null);
    setTouchStart(e.targetTouches[0].clientX);
  };

  const onTouchMove = (e: React.TouchEvent) => {
    setTouchEnd(e.targetTouches[0].clientX);
  };

  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) return;
    
    const distance = touchStart - touchEnd;
    const isLeftSwipe = distance > minSwipeDistance;
    const isRightSwipe = distance < -minSwipeDistance;

    if (isLeftSwipe || isRightSwipe) {
      // Handle swipe - can be used for panel navigation
      console.log('Swipe detected:', isLeftSwipe ? 'left' : 'right');
    }
  };

  return (
    <div
      className={isMobile ? 'mobile-optimized' : ''}
      onTouchStart={onTouchStart}
      onTouchMove={onTouchMove}
      onTouchEnd={onTouchEnd}
      style={{
        touchAction: 'pan-y',
        WebkitTouchCallout: 'none',
        WebkitUserSelect: 'none',
        userSelect: 'none'
      }}
    >
      {children}
      <style>{`
        @media (max-width: ${breakpoint}px) {
          .mobile-optimized {
            font-size: 14px;
          }
          .mobile-optimized button {
            min-height: 44px;
            min-width: 44px;
          }
          .mobile-optimized input {
            font-size: 16px; /* Prevents zoom on iOS */
          }
        }
      `}</style>
    </div>
  );
};

export default MobileOptimized;

