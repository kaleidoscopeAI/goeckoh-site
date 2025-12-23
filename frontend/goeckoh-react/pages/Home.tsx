import React from 'react';
import Button from '../components/Button';
import EchoDemo from '../components/EchoDemo';
import { ArrowRight, Heart, Shield, Activity, Fingerprint } from 'lucide-react';
import { Link } from 'react-router-dom';

const Home: React.FC = () => {
  return (
    <div className="w-full">
      {/* Hero Section */}
      <section className="relative pt-24 pb-32 px-6 overflow-hidden">
        {/* Abstract Background - Pastel Blobs */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full max-w-7xl pointer-events-none">
          <div className="absolute top-20 right-0 w-[500px] h-[500px] bg-brand-peach/30 rounded-full blur-[100px] mix-blend-multiply animate-blob"></div>
          <div className="absolute top-40 left-0 w-[400px] h-[400px] bg-brand-blue/30 rounded-full blur-[100px] mix-blend-multiply animate-blob animation-delay-2000"></div>
          <div className="absolute bottom-20 left-1/2 w-[400px] h-[400px] bg-purple-100 rounded-full blur-[100px] mix-blend-multiply animate-blob animation-delay-4000"></div>
        </div>

        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-16 items-center relative z-10">
          
          {/* Hero Left: Text */}
          <div className="space-y-8">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-brand-blue/30 text-xs font-bold text-brand-accent tracking-wide shadow-sm">
              <span className="w-1.5 h-1.5 rounded-full bg-brand-accent"></span>
              The Neurobiological Mandate for Speech Agency
            </div>
            
            <h1 className="text-5xl md:text-6xl font-extrabold leading-tight tracking-tight text-brand-black">
              <span className="block">Corrected speech,</span>
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-accent to-purple-600 block mt-2">
                in the child’s own voice.
              </span>
            </h1>

            <p className="text-lg text-slate-600 max-w-lg leading-relaxed">
              We leverage biomimetic first-person auditory feedback to repair disrupted corollary discharge signals. It’s not just a tool; it’s a digital exocortex that speaks with your child, grounding cognition and agency in a single, secure system.
            </p>

            <div className="flex flex-wrap gap-4">
              <Button to="/get-app">Get the app</Button>
              <Button to="/science" variant="outline">Read the Research</Button>
            </div>
            
            <div className="pt-6 border-t border-slate-200">
              <p className="text-sm text-slate-500 font-mono italic">
                "I say it. Goeckoh corrects it. I hear myself say it clearly."
              </p>
            </div>
          </div>

          {/* Hero Right: Visual Abstract */}
          <div className="relative flex items-center justify-center min-h-[400px]">
             {/* Echo Bubble Visual */}
            <div className="relative w-80 h-80 rounded-full border border-white/80 bg-white/40 backdrop-blur-xl flex items-center justify-center p-8 shadow-[0_20px_50px_rgba(0,0,0,0.05)] group cursor-default">
              <div className="absolute inset-4 rounded-full border border-brand-blue/20 animate-[ping_3s_linear_infinite]"></div>
              <div className="absolute inset-12 rounded-full border border-brand-peach/30 group-hover:scale-105 transition-transform duration-700"></div>
              
              <div className="text-center space-y-2 relative z-10">
                <p className="text-xs text-slate-400 font-mono tracking-widest uppercase">Self-Agency</p>
                <div className="w-16 h-1 bg-gradient-to-r from-brand-blue to-brand-peach mx-auto rounded-full"></div>
                <p className="text-lg font-bold text-brand-black pt-2">
                  Prediction Error<br/>Minimization
                </p>
              </div>

              {/* Floating Chips */}
              <div className="absolute -bottom-4 bg-white border border-brand-blue/30 text-brand-accent px-4 py-1.5 rounded-full text-xs font-bold shadow-lg">
                Live Neuro-Acoustic Loop
              </div>
            </div>
          </div>

        </div>
      </section>

      {/* Origin Story Section */}
      <section className="py-24 bg-white border-y border-slate-100">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <Heart className="w-12 h-12 text-brand-peach mx-auto mb-6" />
          <h2 className="text-3xl md:text-4xl font-bold text-brand-black mb-8">Not a lab. A home.</h2>
          <div className="space-y-6 text-lg text-slate-600 leading-relaxed font-medium">
            <p>
              This architecture did not originate in a corporate roadmap. It is the result of a two-year, daily effort by a father and self-taught engineer who began building Echo for his autistic son.
            </p>
            <p>
              In the process, he constructed a <strong>Crystalline AGI stack</strong> behind it—a general system for biomimetic modeling of brains and behavior, narrowed and hardened into a therapeutic tool safe enough to sit in front of a child.
            </p>
            <p className="font-bold text-brand-black">
              Its ethics are baked into its origin.
            </p>
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section className="py-24 px-6 relative bg-slate-50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-brand-black mb-4">The First-Person Mirror</h2>
            <p className="text-slate-500 max-w-2xl mx-auto">
              See how the system closes the loop. Goeckoh minimizes prediction error by substituting the error-prone signal with a corrected version in the user's own voice—instantly.
            </p>
          </div>
          <EchoDemo />
        </div>
      </section>

      {/* Scientific Pillars (Teasers) */}
      <section className="py-24 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-end mb-12 gap-6">
             <div>
                <h2 className="text-3xl font-bold text-brand-black mb-2">The Science of Agency</h2>
                <p className="text-slate-500">Grounded in Predictive Coding and Corollary Discharge theory.</p>
             </div>
             <Link to="/science" className="flex items-center gap-2 text-brand-accent hover:text-brand-black transition-colors font-medium">
                Read the full whitepaper <ArrowRight size={16} />
             </Link>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="p-8 rounded-2xl bg-white border border-slate-100 shadow-sm hover:shadow-lg hover:border-brand-blue/50 transition-all group">
              <Activity className="w-8 h-8 text-brand-accent mb-4 group-hover:scale-110 transition-transform" />
              <h3 className="text-xl font-bold text-brand-black mb-3">Corollary Discharge</h3>
              <p className="text-slate-500 text-sm leading-relaxed mb-4">
                We provide a "clean" copy of speech to help the brain suppress N1 auditory responses, a mechanism often disrupted in autism.
              </p>
            </div>
            
            <div className="p-8 rounded-2xl bg-white border border-slate-100 shadow-sm hover:shadow-lg hover:border-brand-peach/50 transition-all group">
              <Fingerprint className="w-8 h-8 text-brand-peach mb-4 group-hover:scale-110 transition-transform" />
              <h3 className="text-xl font-bold text-brand-black mb-3">Identity Congruence</h3>
              <p className="text-slate-500 text-sm leading-relaxed mb-4">
                The "Voice Crystal" maintains 100% acoustic congruence with the child's identity, ensuring the feedback is perceived as "self."
              </p>
            </div>

            <div className="p-8 rounded-2xl bg-white border border-slate-100 shadow-sm hover:shadow-lg hover:border-green-400/50 transition-all group">
              <Shield className="w-8 h-8 text-green-500 mb-4 group-hover:scale-110 transition-transform" />
              <h3 className="text-xl font-bold text-brand-black mb-3">Affective Gating</h3>
              <p className="text-slate-500 text-sm leading-relaxed mb-4">
                Our "Deep Reasoning Core" is strictly gated by the child's emotional state (GCL). Safety comes before instruction.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto bg-brand-black border border-slate-800 rounded-3xl p-12 text-center relative overflow-hidden shadow-2xl">
          <div className="absolute top-0 right-0 w-64 h-64 bg-brand-blue/10 rounded-full blur-[80px]"></div>
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-brand-peach/10 rounded-full blur-[80px]"></div>
          
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-6 relative z-10">
            A safe, private, offline foundation.
          </h2>
          <p className="text-lg text-slate-300 max-w-2xl mx-auto mb-10 relative z-10">
            100% offline processing to ensure privacy. No cloud APIs listening to your home. Built for families, by a family.
          </p>
          <div className="relative z-10 flex justify-center">
             <button className="inline-flex items-center justify-center px-8 py-3 rounded-full font-bold bg-white text-brand-black hover:bg-slate-100 transition-colors">
               Join the Waitlist
             </button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;