import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
}

const Card: React.FC<CardProps> = ({ children, className = '', title }) => {
  return (
    <div className={`bg-white/70 backdrop-blur-md border border-white/50 rounded-2xl p-6 shadow-xl shadow-slate-200/50 ${className}`}>
      {title && (
        <h3 className="text-xl font-bold text-brand-black mb-4">{title}</h3>
      )}
      {children}
    </div>
  );
};

export default Card;