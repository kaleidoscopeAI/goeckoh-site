import React, { useState } from 'react';
import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { Menu, X } from 'lucide-react';

// Pages
import Home from './pages/Home';
import System from './pages/System';
import Parents from './pages/Parents';
import Clinicians from './pages/Clinicians';
import Science from './pages/Science';
import GetApp from './pages/GetApp';
import Dashboard from './pages/Dashboard';

import Button from './components/Button';

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 selection:bg-brand-blue/30">
      <Navbar />
      <main className="min-h-screen pt-20">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/system" element={<System />} />
          <Route path="/parents" element={<Parents />} />
          <Route path="/clinicians" element={<Clinicians />} />
          <Route path="/science" element={<Science />} />
          <Route path="/get-app" element={<GetApp />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
};

const Logo: React.FC = () => (
  <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" className="shadow-sm rounded-xl transition-transform group-hover:scale-105">
    {/* Soft White Background */}
    <rect width="40" height="40" rx="10" fill="#FAFAFA"/>
    <rect x="0.5" y="0.5" width="39" height="39" rx="9.5" stroke="#F1F5F9"/>
    
    {/* Pastel Accents (Blurry Orbs) */}
    <circle cx="30" cy="10" r="12" fill="#BAE6FD" fillOpacity="0.4"/>
    <circle cx="10" cy="30" r="10" fill="#FDBA74" fillOpacity="0.4"/>
    
    {/* Soft blur filter for the orbs to make them look sleek/diffused */}
    <g style={{ filter: 'blur(4px)' }}>
       <circle cx="30" cy="10" r="8" fill="#BAE6FD" fillOpacity="0.6"/>
       <circle cx="10" cy="30" r="8" fill="#FDBA74" fillOpacity="0.6"/>
    </g>

    {/* The Letter G in Black */}
    <path fillRule="evenodd" clipRule="evenodd" d="M20.5 11C16.3579 11 13 14.3579 13 18.5C13 22.6421 16.3579 26 20.5 26C23.2356 26 25.6174 24.5204 26.9248 22.2855C27.1857 21.8394 27.7552 21.6828 28.2013 21.9437C28.6474 22.2046 28.8039 22.7741 28.543 23.2202C26.9365 25.9663 23.9877 27.7857 20.6429 27.7857C15.5144 27.7857 11.3571 23.6285 11.3571 18.5C11.3571 13.3715 15.5144 9.21429 20.6429 9.21429C24.3639 9.21429 27.5619 11.4019 29.0984 14.5982C29.313 15.0445 29.127 15.5821 28.6806 15.7967C28.2343 16.0113 27.6967 15.8253 27.4821 15.3789C26.1738 12.6565 23.4503 10.7925 20.2809 10.7925L20.5 11ZM26.4286 18.5H20.5C19.9477 18.5 19.5 18.9477 19.5 19.5C19.5 20.0523 19.9477 20.5 20.5 20.5H26.4286C26.9809 20.5 27.4286 20.0523 27.4286 19.5C27.4286 18.9477 26.9809 18.5 26.4286 18.5Z" fill="#111827"/>
  </svg>
);

const Navbar: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  const navLinks = [
    { label: 'System', path: '/system' },
    { label: 'For Parents', path: '/parents' },
    { label: 'For Clinicians', path: '/clinicians' },
    { label: 'Science', path: '/science' },
    { label: 'Get the App', path: '/get-app', cta: true },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-100 shadow-sm transition-all duration-300">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between h-20">
          
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <Logo />
            <div className="flex flex-col">
              <span className="text-brand-black font-bold tracking-wide leading-none text-lg">Goeckoh</span>
              <span className="text-[10px] text-slate-500 uppercase tracking-wider leading-none mt-1">Neuro-Acoustic Speech Companion</span>
            </div>
          </Link>

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center gap-8">
            {navLinks.map((link) => (
              link.cta ? (
                <Button key={link.path} to={link.path} className="px-5 py-2 text-sm shadow-md">
                  {link.label}
                </Button>
              ) : (
                <Link 
                  key={link.path} 
                  to={link.path}
                  className={`text-sm font-medium transition-colors hover:text-brand-black ${location.pathname === link.path ? 'text-brand-black font-semibold' : 'text-slate-500'}`}
                >
                  {link.label}
                </Link>
              )
            ))}
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button onClick={() => setIsOpen(!isOpen)} className="text-brand-black">
              {isOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Nav */}
      {isOpen && (
        <div className="md:hidden bg-white border-b border-slate-100 absolute w-full px-6 py-4 flex flex-col gap-4 shadow-xl">
          {navLinks.map((link) => (
            link.cta ? (
              <Button key={link.path} to={link.path} onClick={() => setIsOpen(false)} className="w-full text-center mt-2">
                {link.label}
              </Button>
            ) : (
              <Link 
                key={link.path} 
                to={link.path}
                onClick={() => setIsOpen(false)}
                className={`text-base font-medium py-2 border-b border-slate-50 ${location.pathname === link.path ? 'text-brand-black' : 'text-slate-500'}`}
              >
                {link.label}
              </Link>
            )
          ))}
        </div>
      )}
    </nav>
  );
};

const Footer: React.FC = () => {
  return (
    <footer className="bg-white border-t border-slate-100 pt-16 pb-8 px-6 mt-auto">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
        <div className="text-slate-400 text-sm">
          &copy; {new Date().getFullYear()} Goeckoh Systems.
        </div>
        
        <div className="flex gap-8 text-sm text-slate-500">
          <Link to="/science" className="hover:text-brand-black transition-colors">Technical overview</Link>
          <Link to="/system" className="hover:text-brand-black transition-colors">How it works</Link>
          <Link to="/parents" className="hover:text-brand-black transition-colors">Support for families</Link>
        </div>
      </div>
    </footer>
  );
};

export default App;
