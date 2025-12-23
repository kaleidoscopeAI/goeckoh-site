import React, { useState } from 'react';
import Card from '../components/Card';
import { Mic2, HeartPulse, Cpu, ShieldCheck, ChevronRight, Activity, Database, Sparkles, AudioWaveform } from 'lucide-react';

const System: React.FC = () => {
  return (
    <div className="w-full max-w-7xl mx-auto px-6 py-20">
      <div className="text-center max-w-3xl mx-auto mb-16">
        <h1 className="text-4xl md:text-5xl font-bold text-brand-black mb-6">Echo V4.0 Architecture</h1>
        <p className="text-lg text-slate-600">
          The full loop begins when the child speaks. Echo listens, corrects, and stabilizes speech, 
          then plays it back in the child’s cloned voice. This is not a chatbot; it is a 
          <strong> Crystalline AGI stack</strong> designed for biomimetic modeling of affect and agency.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-20">
        <Card className="relative overflow-hidden group border-slate-100 hover:border-brand-blue/50 transition-all duration-300 hover:shadow-lg cursor-default">
          <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
            <Mic2 size={120} className="text-brand-black" />
          </div>
          <div className="relative z-10">
            <div className="w-12 h-12 bg-brand-blueLight rounded-lg flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Mic2 className="text-brand-accent" size={24} />
            </div>
            <h3 className="text-xl font-bold text-brand-black mb-3">Neuro-Acoustic Mirror</h3>
            <p className="text-slate-500 text-sm leading-relaxed">
              Always-on listening with autism-tuned VAD. It handles segmentation and correction using the "Voice Crystal"—an adaptive cloning engine that evolves with the child via Autopoietic Identity Maintenance (AIM).
            </p>
          </div>
        </Card>

        <Card className="relative overflow-hidden group border-slate-100 hover:border-purple-200 transition-all duration-300 hover:shadow-lg cursor-default">
          <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
            <HeartPulse size={120} className="text-brand-black" />
          </div>
          <div className="relative z-10">
            <div className="w-12 h-12 bg-purple-50 rounded-lg flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <HeartPulse className="text-purple-500" size={24} />
            </div>
            <h3 className="text-xl font-bold text-brand-black mb-3">Crystalline Heart</h3>
            <p className="text-slate-500 text-sm leading-relaxed">
              A 1024-node ODE lattice modeling affective state. It continuously computes a Global Coherence Level (GCL). Low GCL triggers "Self-Preservation Mode," limiting the system to simple, calming interactions.
            </p>
          </div>
        </Card>

        <Card className="relative overflow-hidden group border-slate-100 hover:border-green-200 transition-all duration-300 hover:shadow-lg cursor-default">
          <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
            <Cpu size={120} className="text-brand-black" />
          </div>
          <div className="relative z-10">
            <div className="w-12 h-12 bg-green-50 rounded-lg flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <ShieldCheck className="text-green-500" size={24} />
            </div>
            <h3 className="text-xl font-bold text-brand-black mb-3">Deep Reasoning Core</h3>
            <p className="text-slate-500 text-sm leading-relaxed">
              This reasoning engine is strictly gated. It can only perform complex tasks if the GCL is &ge; 0.9 (Blue Mode). If the child is dysregulated, the AGI functionality is mathematically locked out.
            </p>
          </div>
        </Card>
      </div>

      <div className="bg-white border border-slate-200 rounded-3xl p-8 md:p-12 shadow-sm">
        <h2 className="text-2xl font-bold text-brand-black mb-1">The Self-Correction Loop</h2>
        <p className="text-slate-500 mb-8">Click a step to explore the data flow.</p>
        
        <InteractiveLoopStepper />
      </div>
    </div>
  );
};

const InteractiveLoopStepper: React.FC = () => {
  const [activeStep, setActiveStep] = useState<number>(0);

  const steps = [
    {
      title: "Audio Capture",
      desc: "Device captures audio via 100% offline edge processing. High-sample rate buffers ensure no data is lost during VAD wake-up.",
      icon: Mic2
    },
    {
      title: "Phenotyping & Segmentation",
      desc: "Backend segments utterances and classifies them (Phenotyping Mode) as nonverbal, dysfluent, or clear in < 50ms.",
      icon: Activity
    },
    {
      title: "First-Person Rewriter",
      desc: "The 'First-Person Rewriter' corrects grammar while preserving the 'I/Me' perspective to enforce agency.",
      icon: Sparkles
    },
    {
      title: "Voice Crystal Synthesis",
      desc: "The 'Voice Crystal' synthesizes the output, matching the child's prosody (rhythm/tone) or applying calming regulation.",
      icon: AudioWaveform
    },
    {
      title: "Heart Update",
      desc: "Event is logged to the Crystalline Heart model. GCL (Global Coherence Level) is recalculated based on interaction success.",
      icon: Database
    }
  ];

  return (
    <div className="flex flex-col md:flex-row gap-8">
      {/* Steps List */}
      <div className="flex-1 space-y-4">
        {steps.map((step, index) => {
          const isActive = activeStep === index;
          return (
            <button
              key={index}
              onClick={() => setActiveStep(index)}
              className={`w-full flex items-center gap-4 p-4 rounded-xl border transition-all duration-300 text-left group ${
                isActive 
                  ? 'bg-brand-black border-brand-black text-white shadow-lg scale-[1.02]' 
                  : 'bg-white border-slate-100 hover:border-brand-blue/30 text-slate-500 hover:bg-slate-50'
              }`}
            >
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold border transition-colors ${
                isActive ? 'bg-white/20 border-transparent text-white' : 'bg-slate-100 border-slate-200 text-slate-500'
              }`}>
                {index + 1}
              </div>
              <span className={`font-semibold text-lg flex-1 ${isActive ? 'text-white' : 'text-slate-700'}`}>
                {step.title}
              </span>
              <ChevronRight className={`w-5 h-5 transition-transform ${isActive ? 'text-white translate-x-1' : 'text-slate-300'}`} />
            </button>
          );
        })}
      </div>

      {/* Detail View */}
      <div className="flex-1 bg-slate-50 rounded-2xl border border-slate-200 p-8 flex flex-col justify-center min-h-[300px] relative overflow-hidden">
        <div className="absolute top-0 right-0 p-8 opacity-5 pointer-events-none">
           {React.createElement(steps[activeStep].icon, { size: 200, className: "text-brand-black" })}
        </div>
        
        <div className="relative z-10 animate-[fadeIn_0.3s_ease-out] key={activeStep}"> 
          <div className="w-16 h-16 rounded-2xl bg-white shadow-sm border border-slate-100 flex items-center justify-center mb-6">
            {React.createElement(steps[activeStep].icon, { size: 32, className: "text-brand-accent" })}
          </div>
          <h3 className="text-2xl font-bold text-brand-black mb-4">
            {steps[activeStep].title}
          </h3>
          <p className="text-slate-600 text-lg leading-relaxed">
            {steps[activeStep].desc}
          </p>
        </div>
      </div>
    </div>
  );
};

export default System;