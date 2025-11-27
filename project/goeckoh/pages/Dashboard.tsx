import React from 'react';
import Card from '../components/Card';
import { 
  LineChart, 
  Line, 
  ResponsiveContainer, 
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';

const data = [
  { time: '08:00', value: 0.72 },
  { time: '08:30', value: 0.75 },
  { time: '09:00', value: 0.82 },
  { time: '09:30', value: 0.85 },
  { time: '10:00', value: 0.81 },
  { time: '10:30', value: 0.88 },
  { time: '11:00', value: 0.91 },
  { time: '11:30', value: 0.87 },
  { time: '12:00', value: 0.79 },
  { time: '12:30', value: 0.75 },
  { time: '13:00', value: 0.82 },
  { time: '13:30', value: 0.88 },
  { time: '14:00', value: 0.92 },
  { time: '14:30', value: 0.89 },
  { time: '15:00', value: 0.86 },
  { time: '15:30', value: 0.84 },
  { time: '16:00', value: 0.87 },
  { time: '16:30', value: 0.90 },
  { time: '17:00', value: 0.93 },
  { time: '17:30', value: 0.88 },
];

const Dashboard: React.FC = () => {
  return (
    <div className="w-full max-w-7xl mx-auto px-6 py-12">
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-brand-black">Parent Dashboard</h1>
        <p className="text-slate-500">A live summary of coherence, meltdown risk, and voice profile growth.</p>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
          <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Global Coherence Level</div>
          <div className="text-3xl font-bold text-green-500 mb-1">0.87</div>
          <div className="text-xs text-slate-500">0 = fragmented, 1 = highly coherent</div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
          <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Meltdown Risk</div>
          <div className="text-3xl font-bold text-yellow-500 mb-1">23%</div>
          <div className="text-xs text-slate-500">Probability within the next hour</div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
          <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Voice Profile Score</div>
          <div className="text-3xl font-bold text-brand-accent mb-1">0.91</div>
          <div className="text-xs text-slate-500">Higher = clearer, more stable speech</div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
          <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Utterances today</div>
          <div className="text-3xl font-bold text-brand-black mb-1">143</div>
          <div className="text-xs text-slate-500">Detected by echo loop</div>
        </div>
      </div>

      {/* Chart Section */}
      <Card title="Coherence over time (GCL)">
        <p className="text-sm text-slate-500 mb-6">A smoothed view of your child’s global coherence level throughout the day.</p>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <defs>
                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#22C55E" stopOpacity={0.2}/>
                  <stop offset="95%" stopColor="#22C55E" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <XAxis 
                dataKey="time" 
                stroke="#94a3b8" 
                tick={{fill: '#64748b', fontSize: 12}}
                tickLine={false}
                axisLine={false}
                minTickGap={30}
              />
              <YAxis 
                domain={[0.5, 1]} 
                hide 
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#fff', borderColor: '#e2e8f0', color: '#1e293b', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                itemStyle={{ color: '#22C55E' }}
                formatter={(value: number) => [value, "GCL"]}
              />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#22C55E" 
                strokeWidth={3} 
                dot={false}
                activeDot={{ r: 6, fill: '#22C55E' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
};

export default Dashboard;