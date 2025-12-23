import React, { useState } from 'react';
import Card from '../components/Card';
import { Apple } from 'lucide-react';

const GetApp: React.FC = () => {
  return (
    <div className="w-full max-w-7xl mx-auto px-6 py-20">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
        
        {/* Left Column: Content & Forms */}
        <div>
          <h1 className="text-4xl md:text-5xl font-bold text-brand-black mb-6">Get Goeckoh</h1>
          <p className="text-lg text-slate-600 mb-10">
            The iOS app connects to a secure backend profile and runs the full echo loop: listening, correcting, and playing back speech in the child’s own voice.
          </p>

          <div className="mb-12">
            <button className="flex items-center gap-4 bg-brand-black hover:bg-slate-800 transition-colors rounded-xl px-6 py-3 pr-8 shadow-lg shadow-slate-300">
              <Apple className="w-10 h-10 text-white" />
              <div className="text-left">
                <div className="text-xs text-slate-300">Download on the</div>
                <div className="text-xl font-bold text-white leading-none">App Store (iOS)</div>
              </div>
            </button>
          </div>

          <div className="space-y-8">
            <ParentWaitlistForm />
            <ClinicForm />
          </div>

          <div className="mt-8 pt-8 border-t border-slate-200">
            <p className="text-slate-500">
              Or email us directly at <a href="mailto:care@goeckoh.com" className="text-brand-accent hover:underline">care@goeckoh.com</a>.
            </p>
          </div>
        </div>

        {/* Right Column: Visual Mockup */}
        <div className="hidden lg:flex items-center justify-center relative">
          <div className="relative w-[320px] h-[640px] bg-white rounded-[3rem] border-8 border-slate-100 shadow-2xl flex flex-col items-center p-6 overflow-hidden">
            {/* Screen Content */}
            <div className="absolute top-0 left-0 right-0 h-full w-full bg-slate-50 z-0"></div>
            
            {/* Status Bar */}
            <div className="relative z-10 w-full flex justify-between items-center px-2 pt-2 mb-8">
               <span className="text-[10px] text-slate-500">9:41</span>
               <div className="flex gap-1">
                 <div className="w-3 h-3 bg-slate-300 rounded-full"></div>
                 <div className="w-3 h-3 bg-slate-300 rounded-full"></div>
               </div>
            </div>

            {/* App Branding */}
            <div className="relative z-10 mb-auto mt-10">
              <h2 className="text-2xl font-bold text-brand-black tracking-widest text-center">GOECKOH</h2>
              <div className="mt-2 flex justify-center">
                 <span className="px-3 py-1 bg-green-100 text-green-700 text-[10px] font-bold rounded-full uppercase tracking-wider animate-pulse">Echo Loop Active</span>
              </div>
            </div>

            {/* Visualizer */}
            <div className="relative z-10 w-full h-40 flex items-center justify-center gap-1">
               {[...Array(8)].map((_, i) => (
                 <div key={i} className="w-2 bg-brand-blue rounded-full animate-[pulse_1.5s_ease-in-out_infinite]" style={{ height: `${Math.random() * 40 + 20}px`, animationDelay: `${i * 0.1}s` }}></div>
               ))}
            </div>

            {/* Quote Bubble */}
            <div className="relative z-10 w-full mb-12">
              <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-lg mx-2">
                <p className="text-center text-lg font-medium text-brand-black">"I hear myself say it clearly."</p>
              </div>
            </div>
            
            {/* Bottom Indicator */}
             <div className="relative z-10 w-32 h-1 bg-slate-300 rounded-full mb-2"></div>
          </div>
        </div>
      </div>
    </div>
  );
};

// --- Sub-components for Forms ---

const ParentWaitlistForm: React.FC = () => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [email, setEmail] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setTimeout(() => {
      setIsSubmitting(false);
      setIsSuccess(true);
      setEmail('');
    }, 600);
  };

  return (
    <Card title="For parents & caregivers">
      {isSuccess ? (
        <div className="p-4 bg-green-50 border border-green-100 rounded-lg text-green-700">
          Thank you. We’ll reach out as soon as we expand access.
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm text-slate-500 mb-1">Email address</label>
            <input 
              required
              type="email" 
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-brand-black focus:border-brand-black focus:ring-1 focus:ring-brand-black focus:outline-none transition-all"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-500 mb-1">Child's age (optional)</label>
            <input 
              type="text" 
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-brand-black focus:border-brand-black focus:ring-1 focus:ring-brand-black focus:outline-none transition-all"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-500 mb-1">Anything you’d like us to know (optional)</label>
            <textarea 
              rows={3}
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-brand-black focus:border-brand-black focus:ring-1 focus:ring-brand-black focus:outline-none transition-all"
            />
          </div>
          <button 
            type="submit" 
            disabled={isSubmitting}
            className="px-6 py-2 bg-brand-black hover:bg-slate-800 rounded-full text-white font-medium transition-colors disabled:opacity-50 shadow-lg shadow-slate-200"
          >
            {isSubmitting ? 'Submitting...' : 'Join Waitlist'}
          </button>
        </form>
      )}
    </Card>
  );
};

const ClinicForm: React.FC = () => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [email, setEmail] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setTimeout(() => {
      setIsSubmitting(false);
      setIsSuccess(true);
      setEmail('');
    }, 600);
  };

  return (
    <Card title="For clinics, schools, and research teams">
      {isSuccess ? (
         <div className="p-4 bg-green-50 border border-green-100 rounded-lg text-green-700">
          Thank you. We’ll reach out as soon as we expand access.
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm text-slate-500 mb-1">Clinic / school / research group</label>
            <input 
              required
              type="text" 
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-brand-black focus:border-brand-accent focus:ring-1 focus:ring-brand-accent focus:outline-none transition-all"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-500 mb-1">Contact email</label>
            <input 
              required
              type="email" 
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-brand-black focus:border-brand-accent focus:ring-1 focus:ring-brand-accent focus:outline-none transition-all"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-500 mb-1">Approx. children in program (optional)</label>
            <input 
              type="text" 
              className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-brand-black focus:border-brand-accent focus:ring-1 focus:ring-brand-accent focus:outline-none transition-all"
            />
          </div>
          <button 
            type="submit" 
            disabled={isSubmitting}
            className="px-6 py-2 border border-brand-accent text-brand-accent hover:bg-brand-accent hover:text-white rounded-full font-medium transition-colors disabled:opacity-50"
          >
            {isSubmitting ? 'Submitting...' : 'Request Access'}
          </button>
        </form>
      )}
    </Card>
  );
};

export default GetApp;