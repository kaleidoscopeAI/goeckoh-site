import React from 'react';
import Card from '../components/Card';
import { ClipboardList, Lock, CheckCircle2, Activity, BrainCircuit, BarChart3 } from 'lucide-react';

const Clinicians: React.FC = () => {
  return (
    <div className="w-full max-w-7xl mx-auto px-6 py-20 space-y-16">
      <header className="max-w-3xl space-y-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-blueLight border border-brand-blue/30 text-xs font-bold text-brand-accent tracking-wide">
          For SLPs, neuropsychologists, and researchers
        </div>
        <h1 className="text-4xl md:text-5xl font-bold text-brand-black">For Clinicians: A neuro-acoustic tool that fits inside your existing practice.</h1>
        <p className="text-lg text-slate-600">
          Goeckoh is built to complement, not replace, traditional speech and language therapy. It gives you a way to extend your work into the home, with a device that reinforces self-voice, prediction error reduction, and motor planning between sessions.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <Card title="Clinical use cases" className="h-full">
          <div className="flex items-start gap-4 mb-4">
            <div className="p-3 bg-brand-blueLight rounded-lg">
              <BrainCircuit className="text-brand-accent w-6 h-6" />
            </div>
          </div>
          <ul className="space-y-3 text-slate-600 text-sm">
            <li><strong>Autism with atypical self-voice monitoring.</strong> Provide consistent, high-fidelity self-voice feedback to support agency and N1 suppression.</li>
            <li><strong>Childhood apraxia and motor planning difficulties.</strong> Pair your motor-based drills with a tool that returns a “target” version in the child’s own voice after each attempt.</li>
            <li><strong>Hyperlexia and scripting.</strong> Harness scripts as raw material: Goeckoh cleans them and mirrors them back, helping expand toward flexible, spontaneous language.</li>
            <li><strong>Anxiety and shutdown.</strong> Use the affective gating and calm-mode prompts to keep interaction predictable for children who become overwhelmed by direct instruction.</li>
          </ul>
        </Card>

        <Card title="What the system gives you" className="h-full">
          <div className="flex items-start gap-4 mb-4">
            <div className="p-3 bg-brand-blueLight rounded-lg">
              <BarChart3 className="text-brand-accent w-6 h-6" />
            </div>
          </div>
          <ul className="space-y-3 text-slate-600 text-sm">
            <li><strong>Session logs.</strong> High-level data on number of utterances mirrored, average latency, and time spent in different regulation bands (GCL estimates).</li>
            <li><strong>Echo quality metrics.</strong> Internal scores tracking intelligibility over time (e.g., proportion of heavily vs. lightly corrected words).</li>
            <li><strong>Configurable profiles.</strong> Conservative vs. assertive correction modes to align with each child’s goals and tolerance.</li>
          </ul>
        </Card>

        <Card title="How it fits into your workflow" className="h-full">
          <div className="flex items-start gap-4 mb-4">
            <div className="p-3 bg-brand-blueLight rounded-lg">
              <Activity className="text-brand-accent w-6 h-6" />
            </div>
          </div>
          <ul className="space-y-3 text-slate-600 text-sm">
            <li><strong>Before sessions.</strong> Review the parent dashboard to see engagement patterns and high-frequency phrases; use this to choose meaningful targets.</li>
            <li><strong>During sessions.</strong> Run short “mirror drills”: you model → child imitates → Goeckoh mirrors in the child’s voice; contrast attempted vs. mirrored output to cue awareness.</li>
            <li><strong>Between sessions.</strong> Assign micro-homework: two or three phrases for Goeckoh time each day, with no timers or worksheets. Parents simply ensure the app is available; the system handles the rest.</li>
          </ul>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <Card title="Clinical hooks & research surface" className="h-full">
          <div className="flex items-start gap-4 mb-6">
            <div className="p-3 bg-brand-blueLight rounded-lg">
              <ClipboardList className="text-brand-accent w-6 h-6" />
            </div>
          </div>
          <ul className="space-y-2 text-slate-600 text-sm">
            {[
              "Optional structured trials and probe sessions can be scripted and reviewed with families.",
              "Exportable anonymized metrics (CSV/JSON) for research, progress monitoring, and publication.",
              "Configurable correction policies aligned with verbal behavior, NDBI, or other clinical models.",
              "Clear on/off toggles for experimental features so you can separate routine care from protocol work."
            ].map((item, i) => (
              <li
                key={i}
                className="flex gap-4 items-start p-3 rounded-lg hover:bg-slate-50 transition-all cursor-default group"
              >
                <CheckCircle2 className="w-5 h-5 text-brand-accent mt-0.5 shrink-0 group-hover:scale-110 transition-transform" />
                <span className="group-hover:text-brand-black transition-colors">{item}</span>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Safety, ethics, and evidence alignment" className="h-full">
          <div className="flex items-start gap-4 mb-6">
            <div className="p-3 bg-red-50 rounded-lg">
              <Lock className="text-red-400 w-6 h-6" />
            </div>
          </div>
          <ul className="space-y-2 text-slate-600 text-sm">
            {[
              "Designed around predictive coding and corollary discharge models of speech self-monitoring.",
              "Uses a cloned self-voice to preserve identity congruence, avoiding confusion with character or assistant personas.",
              "Affective gating ensures the reasoning layer never overwhelms a dysregulated child; when coherence drops, the system quietly returns to simple mirroring and co-regulation.",
              "Goeckoh is meant to be a precise, controllable acoustic instrument in your toolkit — one that parents can safely hand to a child at home, while you stay anchored in measurable, neurobiologically grounded outcomes."
            ].map((item, i) => (
              <li
                key={i}
                className="flex gap-4 items-start p-3 rounded-lg hover:bg-red-50/50 transition-all cursor-default group"
              >
                <CheckCircle2 className="w-5 h-5 text-red-400 mt-0.5 shrink-0 group-hover:scale-110 transition-transform" />
                <span className="group-hover:text-brand-black transition-colors">{item}</span>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
};

export default Clinicians;
