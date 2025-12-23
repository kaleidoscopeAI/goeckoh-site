import React from 'react';
import { Link } from 'react-router-dom';

interface ButtonProps {
  children: React.ReactNode;
  to?: string;
  variant?: 'primary' | 'outline' | 'ghost';
  className?: string;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
  disabled?: boolean;
}

const Button: React.FC<ButtonProps> = ({ 
  children, 
  to, 
  variant = 'primary', 
  className = '', 
  onClick,
  type = 'button',
  disabled = false
}) => {
  const baseStyles = "inline-flex items-center justify-center px-6 py-3 rounded-full font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-white disabled:opacity-50 disabled:cursor-not-allowed";
  
  const variants = {
    primary: "bg-brand-black hover:bg-slate-800 text-white shadow-lg shadow-slate-200 focus:ring-slate-900",
    outline: "border border-slate-300 hover:border-brand-accent text-slate-600 hover:text-brand-accent bg-transparent focus:ring-brand-accent",
    ghost: "text-slate-500 hover:text-brand-black hover:bg-slate-100"
  };

  const combinedClasses = `${baseStyles} ${variants[variant]} ${className}`;

  if (to) {
    return (
      <Link to={to} className={combinedClasses}>
        {children}
      </Link>
    );
  }

  return (
    <button type={type} className={combinedClasses} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
};

export default Button;