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
            <p className="text-xs uppercase tracking-[0.2em] text-slate-400 font-semibold">
              Grounded in corollary discharge, predictive coding, and self-voice monitoring.
            </p>
            
            <h1 className="text-5xl md:text-6xl font-extrabold leading-tight tracking-tight text-brand-black">
              <span className="block">Corrected speech,</span>
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-accent to-purple-600 block mt-2">
                in the child’s own voice.
              </span>
            </h1>

            <p className="text-lg text-slate-600 max-w-2xl leading-relaxed">
              Goeckoh listens to imperfect, fragmented, or hard-to-understand speech and instantly returns a corrected version in the child’s exact cloned voice. This closed neuro-acoustic loop is designed to reduce prediction error in the auditory system and help the brain learn: “That clear voice is me.”
            </p>

            <div className="flex flex-col gap-2 text-sm text-slate-600">
              <div className="inline-flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-brand-peach"></span>
                Built for autism, apraxia, and other speech-motor differences
              </div>
              <div className="inline-flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-brand-blue"></span>
                Real-time echo: no scripts, no drills, no flashcards
              </div>
              <div className="inline-flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-brand-black"></span>
                Runs locally on device — no cloud listening, ever
              </div>
            </div>

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
            <p className="mt-8 text-sm text-slate-600 max-w-md text-center mx-auto">
              Goeckoh continuously compares what the child tried to say with a corrected target and feeds back a clean version in their own voice. Over time, this trains the brain’s internal model of “how I sound when I get it right.”
            </p>
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
              Goeckoh did not start as a corporate research project. It began in a living room, built by a father and self-taught engineer for his autistic son who struggled to hear himself clearly.
            </p>
            <p>
              Behind the app is a full <strong>“Crystalline AGI”</strong> stack — a mathematically defined system that models emotional state, speech patterns, and context.
            </p>
            <p className="font-bold text-brand-black">
              But at the surface, it’s simple: a calm, predictable companion that sits on the couch, at the table, or in the car and quietly echoes speech back in the child’s own voice.
            </p>
            <p>
              Every design choice — from offline processing to first-person phrasing — was made for families who need something safe enough to hand to a child and reliable enough to run all day.
            </p>
            <p className="font-semibold text-brand-black">
              Its ethics are baked into its origin: one child, one family, and a system that must never betray their trust.
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
              Goeckoh is not a tutor that waits for pre-set phrases. It is an always-on mirror: whatever the child says, it cleans, corrects, and reflects back in their own voice.
            </p>
            <p className="text-sm uppercase tracking-[0.2em] text-slate-400 mt-4">The loop closes in a few hundred milliseconds:</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            <div className="p-6 bg-white border border-slate-100 rounded-2xl shadow-sm">
              <p className="text-xs font-semibold text-brand-accent mb-2">1. Child speaks</p>
              <p className="text-slate-600 text-sm leading-relaxed">
                Raw speech — including stutters, mispronunciations, and partial words — is captured by an autism-tuned voice activity detector that respects long pauses and quiet, flat prosody.
              </p>
            </div>
            <div className="p-6 bg-white border border-slate-100 rounded-2xl shadow-sm">
              <p className="text-xs font-semibold text-brand-accent mb-2">2. Goeckoh corrects & clones</p>
              <p className="text-slate-600 text-sm leading-relaxed">
                The system transcribes the utterance, repairs grammar and articulation, and re-synthesizes the phrase in the child’s exact voiceprint, preserving natural rhythm where possible.
              </p>
            </div>
            <div className="p-6 bg-white border border-slate-100 rounded-2xl shadow-sm">
              <p className="text-xs font-semibold text-brand-accent mb-2">3. Child hears corrected voice</p>
              <p className="text-slate-600 text-sm leading-relaxed">
                The child hears a clear, fluent version that still “sounds like me,” helping the nervous system align intention, movement, and auditory feedback.
              </p>
            </div>
          </div>

          <EchoDemo />

          <div className="mt-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4 text-sm text-slate-500">
            <div>Typical target: under ~500 ms from speech end to mirrored playback.</div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-slate-700">Global Coherence Level (GCL):</span>
              <span className="text-slate-600">Internal metric estimating how stable and regulated the current emotional state is.</span>
            </div>
          </div>
        </div>
      </section>

      {/* Scientific Pillars (Teasers) */}
      <section className="py-24 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-end mb-12 gap-6">
             <div>
                <h2 className="text-3xl font-bold text-brand-black mb-2">The Science of Agency</h2>
                <p className="text-slate-500">Grounded in predictive coding, corollary discharge, and self-voice identity.</p>
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
                When we speak, the brain predicts the sound of our own voice and dampens the response to it. In autism and some speech-motor conditions, this mechanism can be noisy or disrupted. Goeckoh provides a clean, consistent copy of the intended speech to help the brain re-learn the difference between “my voice” and “external noise,” supporting N1 suppression and self-agency.
              </p>
            </div>
            
            <div className="p-8 rounded-2xl bg-white border border-slate-100 shadow-sm hover:shadow-lg hover:border-brand-peach/50 transition-all group">
              <Fingerprint className="w-8 h-8 text-brand-peach mb-4 group-hover:scale-110 transition-transform" />
              <h3 className="text-xl font-bold text-brand-black mb-3">Identity Congruence</h3>
              <p className="text-slate-500 text-sm leading-relaxed mb-4">
                Feedback only works if it feels like self. Goeckoh’s “Voice Crystal” stores a high-fidelity, evolving model of the child’s voice — timbre, prosody, and micro-inflections — and uses it for all echoes. No generic voice, no character filters: the system’s job is to sound like the child on their best day, every time.
              </p>
            </div>

            <div className="p-8 rounded-2xl bg-white border border-slate-100 shadow-sm hover:shadow-lg hover:border-green-400/50 transition-all group">
              <Shield className="w-8 h-8 text-green-500 mb-4 group-hover:scale-110 transition-transform" />
              <h3 className="text-xl font-bold text-brand-black mb-3">Affective Gating</h3>
              <p className="text-slate-500 text-sm leading-relaxed mb-4">
                A “Deep Reasoning Core” sits behind the mirror, able to generate explanations, social stories, and coaching. But it is strictly gated by the child’s emotional state, estimated by the Crystalline Heart’s Global Coherence Level (GCL). Safety and regulation always come before instruction.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto bg-white border border-slate-200 rounded-3xl p-10 md:p-14 shadow-xl">
          <div className="flex flex-col gap-6">
            <h2 className="text-3xl md:text-4xl font-bold text-brand-black">A safe, private, offline foundation.</h2>
            <ul className="space-y-3 text-slate-600">
              <li>Goeckoh is designed as a closed neuro-acoustic loop, not a cloud service.</li>
              <li><strong>100% local processing</strong> — Speech, emotion estimates, and voice cloning all run on the device. Audio never leaves the home.</li>
              <li><strong>No continuous recording</strong> — The system listens for speech, mirrors it, and keeps only the minimum data needed for on-device learning and progress charts.</li>
              <li><strong>Family-controlled data</strong> — Parents can see, export, or delete logs at any time. There is no external analytics pipeline.</li>
              <li><strong>Built for longevity</strong> — The same companion can grow with the child, adapting its voice model and coaching style over years, not just weeks.</li>
            </ul>
            <p className="text-brand-black font-semibold">Built for families, by a family.</p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
