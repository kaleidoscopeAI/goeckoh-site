import React, { useMemo } from 'react';
import Card from '../components/Card';

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

const Sparkline: React.FC = () => {
  const { pathD, areaD, start, end, min, max } = useMemo(() => {
    const values = data.map((d) => d.value);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const width = 920;
    const height = 260;
    const padX = 12;
    const padY = 18;
    const usableWidth = width - padX * 2;
    const usableHeight = height - padY * 2;
    const step = usableWidth / (data.length - 1);

    const points = data.map((d, idx) => {
      const x = padX + idx * step;
      const ratio = maxVal === minVal ? 0.5 : (d.value - minVal) / (maxVal - minVal);
      const y = padY + (1 - ratio) * usableHeight;
      return { x, y };
    });

    const path = points
      .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`)
      .join(' ');

    const area =
      `M ${padX} ${height - padY} ` +
      points.map((p) => `L ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' ') +
      ` L ${padX + usableWidth} ${height - padY} Z`;

    return {
      pathD: path,
      areaD: area,
      start: points[0],
      end: points[points.length - 1],
      min: minVal,
      max: maxVal,
    };
  }, []);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-400 mb-1">Coherence over time</div>
          <div className="text-lg font-semibold text-brand-black">Global Coherence Level (GCL)</div>
        </div>
        <div className="text-xs text-slate-500">
          <span className="font-semibold text-green-600">Peak:</span> {max.toFixed(2)} &nbsp;|&nbsp; 
          <span className="font-semibold text-slate-700"> Baseline:</span> {min.toFixed(2)}
        </div>
      </div>

      <div className="relative">
        <svg viewBox="0 0 920 260" className="w-full h-[260px]" role="img" aria-label="Coherence over time sparkline">
          <defs>
            <linearGradient id="sparklineGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#22c55e" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#22c55e" stopOpacity="0" />
            </linearGradient>
            <linearGradient id="sparklineStroke" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#22c55e" />
              <stop offset="100%" stopColor="#14b8a6" />
            </linearGradient>
          </defs>

          <path d={areaD} fill="url(#sparklineGradient)" />
          <path d={pathD} fill="none" stroke="url(#sparklineStroke)" strokeWidth="3.5" strokeLinecap="round" />

          <circle cx={start.x} cy={start.y} r="5" fill="#22c55e" stroke="#fff" strokeWidth="2" />
          <circle cx={end.x} cy={end.y} r="5" fill="#0ea5e9" stroke="#fff" strokeWidth="2" />
        </svg>

        <div className="flex justify-between text-xs text-slate-500 mt-3">
          <div>Start: {data[0].time}</div>
          <div>End: {data[data.length - 1].time}</div>
        </div>
      </div>
    </div>
  );
};

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
        <Sparkline />
      </Card>
    </div>
  );
};

export default Dashboard;
