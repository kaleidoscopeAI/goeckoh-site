import React, { useState } from 'react';
import Card from '../components/Card';
import Button from '../components/Button';
import { Sparkles, ShieldCheck, HeartHandshake, WifiOff, Lock, ChevronDown } from 'lucide-react';

const Parents: React.FC = () => {
  return (
    <div className="w-full max-w-7xl mx-auto px-6 py-20">
      <div className="mb-16">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-peachLight border border-brand-peach/30 text-xs font-bold text-brand-accent tracking-wide mb-6">
          Built by a parent, for his son.
        </div>
        <h1 className="text-4xl md:text-5xl font-bold text-brand-black mb-6">Safe enough to sit in front of a child.</h1>
        <p className="text-lg text-slate-600 max-w-2xl">
          Echo is not just a therapy tool; it’s a commitment to safety. Because it was built by a father for his own family, its ethics are baked into its code—not added as an afterthought.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
        <InteractiveCard 
          title="The Parent's Mandate: Absolute Privacy"
          defaultOpen={true}
        >
          <ul className="space-y-6 text-slate-600 pt-2">
            <li className="flex gap-4 p-4 rounded-xl hover:bg-slate-50 transition-colors">
              <div className="p-2 bg-brand-blueLight rounded-lg h-fit">
                <WifiOff className="w-6 h-6 text-brand-accent shrink-0" />
              </div>
              <div>
                <strong className="text-brand-black block mb-1">100% Offline Processing</strong>
                <p className="text-sm text-slate-500">
                  No cloud API costs. No eavesdropping. All voice data is processed locally on your device or dedicated hardware. Your child's voice never leaves your home.
                </p>
              </div>
            </li>
            <li className="flex gap-4 p-4 rounded-xl hover:bg-slate-50 transition-colors">
              <div className="p-2 bg-brand-blueLight rounded-lg h-fit">
                <Lock className="w-6 h-6 text-brand-accent shrink-0" />
              </div>
              <div>
                <strong className="text-brand-black block mb-1">Consent Gating</strong>
                <p className="text-sm text-slate-500">
                  All logging and coaching features are disabled by default. You, the caregiver, must explicitly consent to enable them.
                </p>
              </div>
            </li>
          </ul>
        </InteractiveCard>

        <InteractiveCard 
          title="What Goeckoh does for your child"
          defaultOpen={true}
        >
          <ul className="space-y-2 text-slate-600 pt-2">
             <li className="flex gap-3 items-center p-3 rounded-lg hover:bg-purple-50 transition-colors group">
              <Sparkles className="w-5 h-5 text-purple-300 group-hover:text-purple-500 transition-colors shrink-0" />
              <span className="group-hover:text-purple-900 transition-colors">Reflects their words back clearly in their own voice to build agency.</span>
            </li>
            <li className="flex gap-3 items-center p-3 rounded-lg hover:bg-purple-50 transition-colors group">
              <Sparkles className="w-5 h-5 text-purple-300 group-hover:text-purple-500 transition-colors shrink-0" />
              <span className="group-hover:text-purple-900 transition-colors">Uses first-person language ("I / me / my") to encourage inner dialogue.</span>
            </li>
            <li className="flex gap-3 items-center p-3 rounded-lg hover:bg-purple-50 transition-colors group">
              <Sparkles className="w-5 h-5 text-purple-300 group-hover:text-purple-500 transition-colors shrink-0" />
              <span className="group-hover:text-purple-900 transition-colors">Safety First: System shuts down complex features if overwhelmed (Low GCL).</span>
            </li>
            <li className="flex gap-3 items-center p-3 rounded-lg hover:bg-purple-50 transition-colors group">
              <Sparkles className="w-5 h-5 text-purple-300 group-hover:text-purple-500 transition-colors shrink-0" />
              <span className="group-hover:text-purple-900 transition-colors">Adapts to quiet, monotone, or fragmented speech without punishing pauses.</span>
            </li>
          </ul>
        </InteractiveCard>
      </div>

      <div className="bg-white border border-slate-200 rounded-2xl p-8 md:p-12 flex flex-col md:flex-row items-center justify-between gap-8 shadow-sm">
        <div className="max-w-xl">
          <div className="flex items-center gap-3 mb-4">
             <HeartHandshake className="text-brand-peach w-6 h-6" />
             <h2 className="text-2xl font-bold text-brand-black">Join our family.</h2>
          </div>
          <p className="text-slate-500">
            We are slowly opening access to ensure every family receives the support they need.
          </p>
        </div>
        <Button to="/get-app">Join the Waitlist</Button>
      </div>
    </div>
  );
};

// Interactive Card Component
const InteractiveCard: React.FC<{ title: string; children: React.ReactNode; defaultOpen?: boolean }> = ({ title, children, defaultOpen = false }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className={`bg-white border border-slate-200 rounded-2xl shadow-sm transition-all duration-300 overflow-hidden ${isOpen ? 'ring-2 ring-brand-blue/10' : ''}`}>
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-6 text-left"
      >
        <h3 className="text-xl font-bold text-brand-black">{title}</h3>
        <div className={`p-2 rounded-full bg-slate-50 transition-transform duration-300 ${isOpen ? 'rotate-180 bg-brand-blueLight text-brand-accent' : 'text-slate-400'}`}>
          <ChevronDown size={20} />
        </div>
      </button>
      
      <div className={`transition-all duration-300 ease-in-out ${isOpen ? 'max-h-[500px] opacity-100' : 'max-h-0 opacity-0'}`}>
        <div className="px-6 pb-6 border-t border-slate-50">
          {children}
        </div>
      </div>
    </div>
  );
};

export default Parents;