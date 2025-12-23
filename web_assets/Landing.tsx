import React from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  AlertTriangle,
  Brain,
  Fingerprint,
  HeartPulse,
  Lock,
  Mic2,
  ShieldCheck,
  Sparkles,
  WifiOff,
} from 'lucide-react';

const steps = [
  {
    title: 'Listen & Detect',
    desc: 'Autism-tuned VAD respects long pauses and low-volume attempts while buffering audio locally.',
    icon: Mic2,
  },
  {
    title: 'First-Person Rewrite',
    desc: 'ASR + gentle correction keeps intent intact and mirrors everything in first person to preserve agency.',
    icon: Sparkles,
  },
  {
    title: 'Voice Crystal',
    desc: 'Synthesizes in the child’s cloned timbre with prosody transfer so rhythm and identity are preserved.',
    icon: Fingerprint,
  },
  {
    title: 'Crystalline Heart',
    desc: '1024-node affect lattice computes Global Coherence Level (GCL) and gates how assertive/calming to be.',
    icon: HeartPulse,
  },
  {
    title: 'Safety & Logging',
    desc: 'Offline loop logs utterances, latency, and coherence to profiles for clinicians and parents.',
    icon: ShieldCheck,
  },
];

const sciencePoints = [
  {
    title: 'Corollary Discharge (N1 suppression)',
    body: 'Feeds back a clean self-voice to minimize prediction error and restore self/other distinction.',
  },
  {
    title: 'Predictive Coding',
    body: 'High-confidence self-voice acts as a privileged prior, down-weighting noisy bottom-up signals in ASD.',
  },
  {
    title: 'Phonological Recording Obligation',
    body: 'Every utterance becomes a corrected, first-person template; repetition nudges motor plans toward clarity.',
  },
  {
    title: 'Psychoacoustic Bubble',
    body: 'Bouba/Kiki features and modal voice fields translate speech clarity into calming visuals/controls.',
  },
];

const valueTiles = [
  {
    label: 'Offline-first',
    desc: 'No cloud ASR/TTS required; runs locally for privacy and low latency.',
    icon: WifiOff,
    accent: 'bg-slate-900 text-white',
  },
  {
    label: 'Safety-gated',
    desc: 'Affective GCL locks AGI-style reasoning unless the child is regulated.',
    icon: Lock,
    accent: 'bg-green-50 text-green-700',
  },
  {
    label: 'Identity-congruent',
    desc: 'Voice Crystal preserves timbre and prosody for true self-voice feedback.',
    icon: Fingerprint,
    accent: 'bg-blue-50 text-blue-700',
  },
  {
    label: 'Clinician-ready',
    desc: 'Session metrics, latency, intelligibility, and safety flags logged per child profile.',
    icon: Activity,
    accent: 'bg-amber-50 text-amber-700',
  },
];

const Landing: React.FC = () => {
  return (
    <div className="w-full bg-slate-50 text-slate-900">
      <section className="relative overflow-hidden px-6 pt-24 pb-20">
        <div className="absolute inset-0">
          <div className="absolute -left-10 top-10 h-72 w-72 rounded-full bg-brand-blue/20 blur-[120px]" />
          <div className="absolute right-0 bottom-0 h-80 w-80 rounded-full bg-brand-peach/20 blur-[140px]" />
        </div>

        <div className="relative z-10 mx-auto grid max-w-7xl grid-cols-1 gap-12 lg:grid-cols-[1.1fr,0.9fr] items-center">
          <div className="space-y-6">
            <div className="inline-flex items-center gap-2 rounded-full bg-white px-3 py-1 text-xs font-semibold uppercase tracking-wide text-brand-accent shadow-sm border border-slate-100">
              Neuro-acoustic speech companion · Offline-first
            </div>
            <h1 className="text-4xl md:text-5xl font-black leading-tight text-brand-black">
              Corrected speech, in the child’s own voice—within seconds.
            </h1>
            <p className="text-lg text-slate-600 max-w-2xl">
              Goeckoh listens, rewrites in first person, and plays back in a cloned voice tuned to the child’s prosody. A 1024-node Crystalline Heart computes safety (GCL) so the system stays calming when regulation drops.
            </p>
            <div className="flex flex-wrap gap-4">
              <Link
                to="/get-app"
                className="inline-flex items-center justify-center rounded-full bg-brand-black px-6 py-3 text-white font-semibold shadow-lg shadow-slate-300 transition hover:bg-slate-800"
              >
                Get the app
              </Link>
              <Link
                to="/system"
                className="inline-flex items-center justify-center rounded-full border border-slate-300 px-6 py-3 text-brand-black font-semibold bg-white hover:border-brand-black transition"
              >
                See how it works
              </Link>
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 pt-4">
              {[
                '100% local processing (privacy-safe)',
                'Sub-second loop: attempt → corrected self-voice',
                'GCL safety gating (Flow/Calm/Lock)',
              ].map((item) => (
                <div key={item} className="rounded-2xl bg-white p-4 shadow-sm border border-slate-100 text-sm text-slate-700">
                  {item}
                </div>
              ))}
            </div>
          </div>

          <div className="relative flex items-center justify-center">
            <div className="relative h-full w-full max-w-md rounded-3xl bg-white border border-slate-100 shadow-xl p-6">
              <div className="flex items-center gap-3 text-sm font-semibold text-slate-500 mb-4">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                Live neuro-acoustic loop
              </div>
              <div className="space-y-4">
                {steps.map((step, idx) => (
                  <div
                    key={step.title}
                    className="flex items-start gap-3 rounded-2xl border border-slate-100 bg-slate-50/60 p-3"
                  >
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white shadow-sm border border-slate-100 text-brand-black font-bold">
                      {idx + 1}
                    </div>
                    <div>
                      <p className="font-semibold text-slate-900">{step.title}</p>
                      <p className="text-sm text-slate-600">{step.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-white border-y border-slate-100 py-16 px-6">
        <div className="mx-auto max-w-7xl">
          <div className="flex flex-col gap-8 lg:flex-row lg:items-start lg:justify-between">
            <div className="max-w-3xl space-y-4">
              <p className="text-xs font-bold uppercase tracking-wide text-brand-accent">Built for families & clinics</p>
              <h2 className="text-3xl font-bold text-brand-black">Why Goeckoh works</h2>
              <p className="text-slate-600">
                The system is a corollary-discharge proxy: it replaces noisy output with a corrected self-voice to minimize prediction error. It runs offline for privacy, and every loop updates the affective heart so safety always comes first.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:w-[520px]">
              {valueTiles.map((tile) => (
                <div key={tile.label} className={`rounded-2xl border border-slate-100 p-4 shadow-sm ${tile.accent}`}>
                  <div className="flex items-center gap-3">
                    {React.createElement(tile.icon, { size: 18 })}
                    <p className="text-sm font-semibold">{tile.label}</p>
                  </div>
                  <p className="mt-2 text-sm text-inherit opacity-80">{tile.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="py-16 px-6">
        <div className="mx-auto max-w-7xl">
          <div className="flex flex-col gap-3 mb-10">
            <p className="text-xs font-bold uppercase tracking-wide text-slate-500">Evidence</p>
            <h3 className="text-3xl font-bold text-brand-black">Science pillars at a glance</h3>
            <p className="text-slate-600 max-w-3xl">
              Drawn from corollary discharge, predictive coding, and speech-motor control literature—aligned with the Echo V4.0 architecture described in our docs.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            {sciencePoints.map((point) => (
              <div key={point.title} className="rounded-2xl border border-slate-100 bg-white p-6 shadow-sm">
                <p className="text-sm font-semibold text-brand-black">{point.title}</p>
                <p className="mt-2 text-sm text-slate-600 leading-relaxed">{point.body}</p>
              </div>
            ))}
          </div>

          <div className="mt-10 grid gap-6 rounded-3xl border border-slate-100 bg-slate-900 p-8 text-slate-100 shadow-lg md:grid-cols-[1.2fr,0.8fr]">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-emerald-300">Origin</p>
              <h4 className="mt-2 text-2xl font-bold text-white">Built by a dad for his autistic son</h4>
              <p className="mt-3 text-sm text-slate-200">
                Two years of daily, hands-on engineering produced the Crystalline AGI stack behind Echo. The ethics, safety gating, and offline posture are baked into the origin story—built to be safe enough for a child’s bedroom.
              </p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
              <div className="flex items-center gap-3 text-sm font-semibold text-amber-200">
                <AlertTriangle size={18} />
                Safety modes
              </div>
              <ul className="mt-4 space-y-3 text-sm text-slate-200">
                <li>• Red (GCL &lt; 0.5): Self-preservation; only calming affirmations.</li>
                <li>• Yellow (0.5–0.7): Low-risk automation; no new learning tasks.</li>
                <li>• Green (0.7–0.9): Full speech self-correction loop active.</li>
                <li>• Blue (≥ 0.9): Deep reasoning unlocked for complex dialogue.</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-white border-t border-slate-100 px-6 py-16">
        <div className="mx-auto max-w-7xl grid gap-12 lg:grid-cols-[1.1fr,0.9fr]">
          <div className="space-y-4">
            <p className="text-xs font-bold uppercase tracking-wide text-brand-accent">Deployment</p>
            <h3 className="text-3xl font-bold text-brand-black">Ready for homes, clinics, and research</h3>
            <ul className="space-y-3 text-slate-600">
              <li className="flex gap-3">
                <Brain className="mt-1 h-5 w-5 text-brand-accent" />
                <div>
                  <p className="font-semibold text-slate-900">API + GUI + Mobile</p>
                  <p className="text-sm">REST endpoints, PySide desktop GUI, and an iOS app that connects to secure profiles.</p>
                </div>
              </li>
              <li className="flex gap-3">
                <ShieldCheck className="mt-1 h-5 w-5 text-green-500" />
                <div>
                  <p className="font-semibold text-slate-900">Data hygiene</p>
                  <p className="text-sm">Session logs and profiles stay local; no cloud speech APIs are required.</p>
                </div>
              </li>
              <li className="flex gap-3">
                <HeartPulse className="mt-1 h-5 w-5 text-purple-500" />
                <div>
                  <p className="font-semibold text-slate-900">Monitoring</p>
                  <p className="text-sm">Metrics for latency, intelligibility, GCL, and safety flags to review between sessions.</p>
                </div>
              </li>
            </ul>
          </div>

          <div className="rounded-3xl border border-slate-100 bg-slate-900 p-8 text-slate-100 shadow-xl">
            <p className="text-xs font-semibold uppercase tracking-wide text-emerald-300">Get started</p>
            <h4 className="mt-3 text-2xl font-bold text-white">Join the early access list</h4>
            <p className="mt-2 text-sm text-slate-200">
              Tell us where to send setup instructions and onboarding materials for your clinic, school, or home.
            </p>

            <form className="mt-6 space-y-4">
              <div>
                <label className="text-xs text-slate-300">Email</label>
                <input
                  type="email"
                  required
                  placeholder="you@example.com"
                  className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-800 px-4 py-3 text-white placeholder:text-slate-500 focus:border-emerald-300 focus:outline-none"
                />
              </div>
              <div>
                <label className="text-xs text-slate-300">Who are you?</label>
                <select
                  className="mt-1 w-full rounded-xl border border-slate-700 bg-slate-800 px-4 py-3 text-white focus:border-emerald-300 focus:outline-none"
                  defaultValue="parent"
                >
                  <option value="parent">Parent / caregiver</option>
                  <option value="clinician">Clinician / SLP</option>
                  <option value="researcher">Research team</option>
                  <option value="school">School / program</option>
                </select>
              </div>
              <button
                type="submit"
                className="w-full rounded-full bg-emerald-400 px-6 py-3 text-center text-slate-900 font-semibold shadow-lg shadow-emerald-200 transition hover:bg-emerald-300"
              >
                Request access
              </button>
            </form>
            <p className="mt-4 text-xs text-slate-400">
              We keep profiles local by default. No cloud speech APIs or third-party data sharing.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Landing;
