import React, { useState } from 'react';
import Card from '../components/Card';
import Button from '../components/Button';
import { Sparkles, ShieldCheck, HeartHandshake, BarChart3, Activity, ChevronDown } from 'lucide-react';

const Parents: React.FC = () => {
  return (
    <div className="w-full max-w-7xl mx-auto px-6 py-20 space-y-16">
      <header className="space-y-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-peachLight border border-brand-peach/30 text-xs font-bold text-brand-accent tracking-wide">
          Built by a parent, for his son.
        </div>
        <h1 className="text-4xl md:text-5xl font-bold text-brand-black">For Parents: A calm, concrete way to support speech at home.</h1>
        <p className="text-lg text-slate-600 max-w-3xl">
          Goeckoh is designed for real life: busy mornings, car rides, meltdowns, and quiet evenings. It does not replace speech therapy or school supports; it extends them into the moments when your child is most available to learn — when they are relaxed, playing, or simply talking to themselves.
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <InteractiveCard title="What you do at home" defaultOpen>
          <ul className="space-y-4 text-slate-600 pt-2">
            <li className="flex gap-4 p-3 rounded-xl hover:bg-slate-50 transition-colors">
              <div className="p-2 bg-brand-blueLight rounded-lg h-fit">
                <ShieldCheck className="w-6 h-6 text-brand-accent shrink-0" />
              </div>
              <div>
                <strong className="text-brand-black block mb-1">Press one button and talk.</strong>
                <p className="text-sm text-slate-500">
                  Open the app, hand it to your child, or keep it nearby as they move around the house.
                </p>
              </div>
            </li>
            <li className="flex gap-4 p-3 rounded-xl hover:bg-slate-50 transition-colors">
              <div className="p-2 bg-brand-blueLight rounded-lg h-fit">
                <Sparkles className="w-6 h-6 text-brand-accent shrink-0" />
              </div>
              <div>
                <strong className="text-brand-black block mb-1">Let them explore.</strong>
                <p className="text-sm text-slate-500">
                  They can repeat favorite words, scripts, or new phrases. Goeckoh mirrors each attempt in their own voice.
                </p>
              </div>
            </li>
            <li className="flex gap-4 p-3 rounded-xl hover:bg-slate-50 transition-colors">
              <div className="p-2 bg-brand-blueLight rounded-lg h-fit">
                <HeartHandshake className="w-6 h-6 text-brand-accent shrink-0" />
              </div>
              <div>
                <strong className="text-brand-black block mb-1">Use tiny prompts, not pressure.</strong>
                <p className="text-sm text-slate-500">
                  Short invitations like “Tell Goeckoh what you want” or “Show Goeckoh your new word” are enough.
                </p>
              </div>
            </li>
          </ul>
        </InteractiveCard>

        <InteractiveCard title="What Goeckoh does in the background" defaultOpen>
          <ul className="space-y-3 text-slate-600 pt-2">
            <li className="flex gap-3 items-start p-3 rounded-lg hover:bg-purple-50 transition-colors group">
              <Sparkles className="w-5 h-5 text-purple-300 group-hover:text-purple-500 transition-colors shrink-0 mt-1" />
              <span className="group-hover:text-purple-900 transition-colors text-sm">
                Catches imperfect speech — even when it’s quiet, flat, or partially formed.
              </span>
            </li>
            <li className="flex gap-3 items-start p-3 rounded-lg hover:bg-purple-50 transition-colors group">
              <Sparkles className="w-5 h-5 text-purple-300 group-hover:text-purple-500 transition-colors shrink-0 mt-1" />
              <span className="group-hover:text-purple-900 transition-colors text-sm">
                Cleans it up — repairing sounds, order, and clarity while keeping the child’s natural rhythm.
              </span>
            </li>
            <li className="flex gap-3 items-start p-3 rounded-lg hover:bg-purple-50 transition-colors group">
              <Sparkles className="w-5 h-5 text-purple-300 group-hover:text-purple-500 transition-colors shrink-0 mt-1" />
              <span className="group-hover:text-purple-900 transition-colors text-sm">
                Plays it back in their own voice — so the brain can line up “what I meant,” “what I said,” and “what I heard.”
              </span>
            </li>
          </ul>
        </InteractiveCard>
      </div>

      <section className="space-y-4">
        <h2 className="text-2xl md:text-3xl font-bold text-brand-black">Changes many parents look for</h2>
        <p className="text-slate-600 max-w-3xl">
          These are the kinds of shifts Goeckoh is designed to support over weeks and months.
          Every child is different, but the goal is to make good moments more frequent.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <Card className="h-full">
            <div className="flex items-center gap-3 mb-3">
              <Sparkles className="w-5 h-5 text-brand-accent" />
              <h3 className="font-semibold text-brand-black text-sm uppercase tracking-wide">More attempts to speak</h3>
            </div>
            <p className="text-sm text-slate-500">
              The mirror makes talking feel rewarding and “worth the effort,” especially for kids who usually give up quickly.
            </p>
          </Card>
          <Card className="h-full">
            <div className="flex items-center gap-3 mb-3">
              <Activity className="w-5 h-5 text-brand-accent" />
              <h3 className="font-semibold text-brand-black text-sm uppercase tracking-wide">Clearer consonants and endings</h3>
            </div>
            <p className="text-sm text-slate-500">
              Hearing a crisp version of their own words helps the brain notice what was missing and adjust future attempts.
            </p>
          </Card>
          <Card className="h-full">
            <div className="flex items-center gap-3 mb-3">
              <BarChart3 className="w-5 h-5 text-brand-accent" />
              <h3 className="font-semibold text-brand-black text-sm uppercase tracking-wide">Longer phrases</h3>
            </div>
            <p className="text-sm text-slate-500">
              Children often begin by echoing single words, then spontaneously try full sentences as the system models them back.
            </p>
          </Card>
          <Card className="h-full">
            <div className="flex items-center gap-3 mb-3">
              <HeartHandshake className="w-5 h-5 text-brand-accent" />
              <h3 className="font-semibold text-brand-black text-sm uppercase tracking-wide">More “That’s me” moments</h3>
            </div>
            <p className="text-sm text-slate-500">
              Kids listen to their mirrored voice with curiosity — pointing, smiling, or repeating themselves as they realize “I can sound like that.”
            </p>
          </Card>
          <Card className="h-full">
            <div className="flex items-center gap-3 mb-3">
              <ShieldCheck className="w-5 h-5 text-brand-accent" />
              <h3 className="font-semibold text-brand-black text-sm uppercase tracking-wide">Less conflict around speaking practice</h3>
            </div>
            <p className="text-sm text-slate-500">
              Because Goeckoh just “lives” in the environment, practice happens naturally in play, rather than as a separate, stressful task.
            </p>
          </Card>
        </div>
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Card className="h-full">
          <div className="flex items-center gap-3 mb-4">
            <BarChart3 className="w-6 h-6 text-brand-accent" />
            <h2 className="text-xl font-bold text-brand-black">How you stay in control</h2>
          </div>
          <ul className="space-y-3 text-slate-600 text-sm">
            <li>Session summaries show how long your child engaged with the mirror and how many successful echoes occurred.</li>
            <li>Simple progress charts highlight gradual improvements in clarity and phrase length over time.</li>
            <li>Privacy controls let you choose how much data to keep on-device and when to wipe it. Everything is local by default.</li>
          </ul>
        </Card>

        <Card className="h-full">
          <div className="flex items-center gap-3 mb-4">
            <HeartHandshake className="w-6 h-6 text-brand-peach" />
            <h2 className="text-xl font-bold text-brand-black">Why this matters</h2>
          </div>
          <p className="text-slate-600 text-sm leading-relaxed">
            Goeckoh’s job is to quietly make the good moments more frequent — and to give your child a stable, familiar voice that shows them what they’re capable of.
          </p>
        </Card>
      </section>

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
      
      <div className={`transition-all duration-300 ease-in-out ${isOpen ? 'max-h-[600px] opacity-100' : 'max-h-0 opacity-0'}`}>
        <div className="px-6 pb-6 border-t border-slate-50">
          {children}
        </div>
      </div>
    </div>
  );
};

export default Parents;
