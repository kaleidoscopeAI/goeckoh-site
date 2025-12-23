import React from 'react';
import Card from '../components/Card';
import { ClipboardList, Lock, CheckCircle2 } from 'lucide-react';

const Clinicians: React.FC = () => {
  return (
    <div className="w-full max-w-7xl mx-auto px-6 py-20">
      <div className="mb-16">
        <h1 className="text-4xl md:text-5xl font-bold text-brand-black mb-6">For clinicians</h1>
        <p className="text-lg text-slate-600 max-w-2xl">
          Goeckoh extends clinical work into daily environments. It is not a replacement for therapy, but a powerful bridge that reinforces therapeutic goals in naturalistic settings.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <Card title="Clinical hooks" className="h-full">
          <div className="flex items-start gap-4 mb-6">
            <div className="p-3 bg-brand-blueLight rounded-lg">
              <ClipboardList className="text-brand-accent w-6 h-6" />
            </div>
          </div>
          <ul className="space-y-2 text-slate-600">
            {[
              "Optional structured trials and probe sessions can be triggered remotely.",
              "Exportable anonymized metrics (CSV/JSON) for research and progress reports.",
              "Configurable correction policies aligned with specific clinical models (e.g., verbal behavior, NDBI)."
            ].map((item, i) => (
              <li key={i} className="flex gap-4 items-start p-3 rounded-lg hover:bg-slate-50 transition-all cursor-default group">
                <CheckCircle2 className="w-5 h-5 text-brand-accent mt-0.5 shrink-0 group-hover:scale-110 transition-transform" />
                <span className="group-hover:text-brand-black transition-colors">{item}</span>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Safety & boundaries" className="h-full">
          <div className="flex items-start gap-4 mb-6">
            <div className="p-3 bg-red-50 rounded-lg">
              <Lock className="text-red-400 w-6 h-6" />
            </div>
          </div>
          <ul className="space-y-2 text-slate-600">
             {[
               "Goeckoh never pretends to be the child; it only echoes their speech to reinforce agency.",
               "AGI and reasoning components are strictly gated by the affective state model and clinician/parent settings.",
               "All logs are auditable; there are no hidden reinforcement loops."
             ].map((item, i) => (
              <li key={i} className="flex gap-4 items-start p-3 rounded-lg hover:bg-red-50/50 transition-all cursor-default group">
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