import React, { useState } from 'react';
import Card from '../components/Card';
import { BrainCircuit, Activity, ShieldCheck, Lock, Unlock, Zap, AlertTriangle } from 'lucide-react';

const Science: React.FC = () => {
  return (
    <div className="w-full max-w-7xl mx-auto px-6 py-20">
      {/* Header */}
      <div className="mb-20 max-w-4xl">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-50 border border-purple-200 text-xs font-bold text-purple-600 tracking-wide mb-6">
          Research & Foundations
        </div>
        <h1 className="text-4xl md:text-6xl font-bold text-brand-black mb-8 leading-tight">
          The Neurobiological Mandate.
        </h1>
        <p className="text-xl text-slate-600 leading-relaxed">
          The foundation of language acquisition hinges on intact neural feedback mechanisms. Echo V4.0 is architected to intervene in these mechanisms, offering a computationally precise feedback loop designed to initiate speech internalization.
        </p>
      </div>

      {/* Section I */}
      <div className="mb-24">
        <h2 className="text-2xl font-bold text-brand-black mb-8 flex items-center gap-3">
          <span className="text-brand-accent">I.</span> The Auditory-Motor Feedback Loop
        </h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          <div className="space-y-6 text-slate-600">
            <h3 className="text-xl font-semibold text-brand-black">Corollary Discharge & Prediction Error</h3>
            <p>
              Efficient vocal production requires the brain to continuously predict the sensory consequences of its motor commands, a function mediated by the <strong>Corollary Discharge (CD)</strong>. During vocalization, this mechanism suppresses the N1 component of the auditory event-related potential in the auditory cortex.
            </p>
            <p>
              This suppression filters self-generated acoustic input, preventing it from being interpreted as external noise and establishing self-agency over speech. In conditions like autism, this loop can be disrupted.
            </p>
            <div className="p-6 bg-brand-blueLight border-l-4 border-brand-accent rounded-r-xl">
              <p className="text-sm italic text-slate-700 font-medium">
                "Echo substitutes the error-prone signal with a corrected version in the user's own voice. This acts as a Corollary Discharge Proxy, minimizing Prediction Error and forcing the auditory-motor system to update its model."
              </p>
            </div>
          </div>
          
          <div className="space-y-6 text-slate-600">
            <h3 className="text-xl font-semibold text-brand-black">Predictive Coding & Atypical Priors</h3>
            <p>
              Predictive Coding theory holds that the brain constantly adjusts internal models based on sensory feedback. In ASD, atypical priors and prediction errors arise due to impaired audio-vocal integration.
            </p>
            <p>
              Echo’s <strong>Voice Crystal</strong> and <strong>Prosody Transfer</strong> correct both content and delivery in the user's own voice. This provides a high-confidence input that engages internal error correction circuits in the pMTG (posterior Middle Temporal Gyrus).
            </p>
          </div>
        </div>
      </div>

      {/* Section II */}
      <div className="mb-24">
        <h2 className="text-2xl font-bold text-brand-black mb-8 flex items-center gap-3">
          <span className="text-brand-peach">II.</span> Validation of the Self-Correction Hypothesis
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <Card title="Phonological Recoding" className="hover:border-brand-peach/50 transition-colors cursor-default">
            <p className="text-sm text-slate-500">
              Echo externalizes and corrects speech, then feeds it back in a congruent, first-person form. This induces a <strong>Phonological Recoding Obligation (PRO)</strong>—forcing the brain to update based on accurate input.
            </p>
          </Card>
          <Card title="Dynamic Style Selection" className="hover:border-brand-peach/50 transition-colors cursor-default">
            <p className="text-sm text-slate-500">
              Prosody Transfer preserves the child’s speech rhythm. If stress is detected, the system selects a calming style, creating a dual-loop system: linguistic minimization and affective co-regulation.
            </p>
          </Card>
          <Card title="Phenotyping Mode" className="hover:border-brand-peach/50 transition-colors cursor-default">
            <p className="text-sm text-slate-500">
              We classify utterances in real-time. Dysfluent attempts trigger the linguistic loop; nonverbal vocalizations trigger affective intervention; monosyllabic delays trigger supportive pacing.
            </p>
          </Card>
        </div>
      </div>

      {/* Section III - Interactive GCL */}
      <div className="mb-24">
        <h2 className="text-2xl font-bold text-brand-black mb-8 flex items-center gap-3">
          <span className="text-green-500">III.</span> The Echo Architecture V4.0
        </h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-start">
          <div>
            <h3 className="text-xl font-bold text-brand-black mb-4">The Crystalline Heart: Affective Modeling</h3>
            <p className="text-slate-600 mb-6">
              Echo models internal affect as a 1024-node ODE (Ordinary Differential Equation) lattice. It produces a metric called <strong>Global Coherence Level (GCL)</strong>. Low GCL predicts overload and triggers calming actions like low-prosody inner speech.
            </p>
            <h3 className="text-xl font-bold text-brand-black mb-4">Autopoietic Identity Maintenance (AIM)</h3>
            <p className="text-slate-600 mb-6">
              The Voice Crystal evolves with the user. New embeddings are integrated only if they reflect successful speech, monitored via Congruence Drift ($\Delta$) to ensure continuity of voice identity.
            </p>
          </div>
          
          {/* Interactive GCL Table Component */}
          <InteractiveGCLTable />
        </div>
      </div>

      {/* Section IV */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-brand-black mb-8 flex items-center gap-3">
          <span className="text-slate-400">IV.</span> Therapeutic Extensions
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="p-6 border border-slate-200 bg-white rounded-xl shadow-sm hover:shadow-md transition-shadow">
             <h3 className="text-lg font-bold text-brand-black mb-2 flex items-center gap-2">
               <BrainCircuit className="w-5 h-5 text-brand-accent" />
               TBI (Traumatic Brain Injury)
             </h3>
             <p className="text-slate-500 text-sm leading-relaxed">
               TBI disrupts predictive white matter tracts. Echo addresses this through Prediction Error Minimization, acting as a flawless prediction signal to guide the brain toward healthy attractors.
             </p>
          </div>
          <div className="p-6 border border-slate-200 bg-white rounded-xl shadow-sm hover:shadow-md transition-shadow">
             <h3 className="text-lg font-bold text-brand-black mb-2 flex items-center gap-2">
               <Activity className="w-5 h-5 text-brand-peach" />
               Alzheimer’s Disease
             </h3>
             <p className="text-slate-500 text-sm leading-relaxed">
               AD patients exhibit reduced Speaking-Induced Suppression. Echo provides a Corollary Discharge Proxy to maintain the fidelity of the auditory-motor loop and preserve self-agency.
             </p>
          </div>
        </div>
      </div>

       <div className="text-center border-t border-slate-100 pt-12 mt-16">
        <p className="text-slate-500 italic max-w-2xl mx-auto font-serif">
          "Echo is not assistive tech. It is a digital exocortex that speaks with the user in their own voice—grounding cognition, emotion, and agency in a single system."
        </p>
      </div>

    </div>
  );
};

// --- Interactive GCL Component ---

const InteractiveGCLTable: React.FC = () => {
  const [activeLevel, setActiveLevel] = useState<'red' | 'yellow' | 'green' | 'blue'>('green');

  const levels = {
    red: {
      range: "GCL < 0.5",
      mode: "Red / Self-Preservation",
      action: "Calming, Internal Only",
      agiStatus: "Locked",
      desc: "System detects dysregulation or overload. All complex processing is suspended. Output shifts to simple, rhythmic, low-prosody affirmations to restore vagal tone.",
      icon: AlertTriangle,
      color: "text-red-500",
      bg: "bg-red-50",
      border: "border-red-200",
      ring: "ring-red-100"
    },
    yellow: {
      range: "0.5 - 0.7",
      mode: "Yellow / Internal Focus",
      action: "Low-risk Automation",
      agiStatus: "Restricted",
      desc: "User is stable but sensitive. New learning tasks are paused. System focuses on reinforcing known successes and maintaining congruence.",
      icon: ShieldCheck,
      color: "text-yellow-600",
      bg: "bg-yellow-50",
      border: "border-yellow-200",
      ring: "ring-yellow-100"
    },
    green: {
      range: "0.7 - 0.9",
      mode: "Green / Learning",
      action: "Core SCH + Research",
      agiStatus: "Active",
      desc: "Optimal coherence. The full Self-Correction Hypothesis loop is active. System introduces new vocabulary and linguistic challenges.",
      icon: Activity,
      color: "text-green-600",
      bg: "bg-green-50",
      border: "border-green-200",
      ring: "ring-green-100"
    },
    blue: {
      range: "GCL ≥ 0.9",
      mode: "Blue / Executive",
      action: "Complex Automation",
      agiStatus: "Unlocked",
      desc: "High coherence / Flow state. The Deep Reasoning Core is fully unlocked, allowing for complex multi-turn conversations and abstract reasoning tasks.",
      icon: Zap,
      color: "text-brand-accent",
      bg: "bg-brand-blueLight",
      border: "border-brand-blue/30",
      ring: "ring-brand-blueLight"
    }
  };

  const current = levels[activeLevel];
  const Icon = current.icon;

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-8 shadow-sm h-full flex flex-col">
      <h3 className="text-lg font-bold text-brand-black mb-6 border-b border-slate-100 pb-4 flex justify-between items-center">
        <span>Logic Gating by GCL</span>
        <span className="text-xs font-normal text-slate-400">Click to inspect modes</span>
      </h3>
      
      {/* Table Selection */}
      <div className="space-y-3 font-mono text-sm mb-8">
        {(Object.keys(levels) as Array<keyof typeof levels>).map((key) => {
          const lvl = levels[key];
          const isActive = activeLevel === key;
          return (
            <button
              key={key}
              onClick={() => setActiveLevel(key)}
              className={`w-full flex justify-between items-center p-4 rounded-xl border transition-all duration-200 ${
                isActive 
                  ? `${lvl.bg} ${lvl.border} shadow-sm scale-[1.02]` 
                  : 'bg-white border-transparent hover:bg-slate-50 text-slate-400'
              }`}
            >
              <span className={`font-bold ${isActive ? lvl.color : 'text-slate-500'}`}>{lvl.range}</span>
              <span className={`${isActive ? 'text-slate-700 font-semibold' : ''}`}>{lvl.mode.split('/')[0]}</span>
            </button>
          );
        })}
      </div>

      {/* Details Panel */}
      <div className={`mt-auto p-6 rounded-xl border transition-all duration-300 ${current.bg} ${current.border}`}>
        <div className="flex items-center gap-3 mb-3">
          <Icon className={`w-5 h-5 ${current.color}`} />
          <h4 className={`font-bold ${current.color}`}>{current.mode}</h4>
        </div>
        
        <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-slate-500 mb-4 border-b border-black/5 pb-2">
          <span>Action: {current.action}</span>
          <span className="flex items-center gap-1">
            AGI: {current.agiStatus === 'Locked' ? <Lock size={10} /> : <Unlock size={10} />} {current.agiStatus}
          </span>
        </div>

        <p className="text-sm text-slate-700 leading-relaxed">
          {current.desc}
        </p>
      </div>
    </div>
  );
};

export default Science;