import React, { useState, useRef, useEffect } from 'react';
import { Mic, Play, Activity, AlertCircle } from 'lucide-react';
import { DemoPhase } from '../types';

const EchoDemo: React.FC = () => {
  const [phase, setPhase] = useState<DemoPhase>(DemoPhase.IDLE);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [startTime, setStartTime] = useState<number>(0);
  const [audioError, setAudioError] = useState<boolean>(false);

  // References for audio elements
  const inputAudioRef = useRef<HTMLAudioElement | null>(null);
  const outputAudioRef = useRef<HTMLAudioElement | null>(null);

  // GCL Values based on phase
  const getGCL = () => {
    switch (phase) {
      case DemoPhase.IDLE: return '0.86';
      case DemoPhase.LISTENING: return '0.80';
      case DemoPhase.PROCESSING: return '0.78';
      case DemoPhase.PLAYBACK: return '0.89';
      default: return '0.86';
    }
  };

  const handleStartDemo = () => {
    if (phase !== DemoPhase.IDLE) return;
    setAudioError(false);
    setPhase(DemoPhase.LISTENING);
    
    if (inputAudioRef.current) {
      inputAudioRef.current.currentTime = 0;
      inputAudioRef.current.play().catch(() => {
        setAudioError(true);
        setPhase(DemoPhase.IDLE);
      });
    }
  };

  const handleInputEnded = () => {
    setPhase(DemoPhase.PROCESSING);
    setStartTime(performance.now());
    
    // Simulate processing time
    const processingTime = Math.floor(Math.random() * (350 - 220 + 1) + 220);
    
    setTimeout(() => {
      const end = performance.now();
      setLatencyMs(processingTime); 
      
      setPhase(DemoPhase.PLAYBACK);
      if (outputAudioRef.current) {
        outputAudioRef.current.currentTime = 0;
        outputAudioRef.current.play().catch(() => {
          setAudioError(true);
          setPhase(DemoPhase.IDLE);
        });
      }
    }, processingTime);
  };

  const handleOutputEnded = () => {
    setPhase(DemoPhase.IDLE);
  };

  return (
    <div className="w-full max-w-4xl mx-auto mt-12 bg-white/60 backdrop-blur-xl border border-white/50 shadow-2xl shadow-brand-blue/10 rounded-3xl p-6 md:p-10 relative overflow-hidden">
      {/* Hidden Audio Elements */}
      <audio 
        ref={inputAudioRef} 
        src="/audio/child_input_1.m4a" 
        onEnded={handleInputEnded}
        onError={() => setAudioError(true)}
      />
      <audio 
        ref={outputAudioRef} 
        src="/audio/echo_output_1.m4a" 
        onEnded={handleOutputEnded}
        onError={() => setAudioError(true)}
      />

      <div className="absolute top-0 right-0 p-6 flex gap-4 text-sm font-mono">
        <div className="flex flex-col items-end">
          <span className="text-slate-400">Demo latency:</span>
          <span className="text-brand-accent font-bold">{latencyMs !== null ? `${latencyMs} ms` : '—'}</span>
        </div>
        <div className="flex flex-col items-end">
          <span className="text-slate-400">Demo GCL:</span>
          <span className={`font-bold transition-colors duration-300 ${phase === DemoPhase.PLAYBACK ? 'text-green-500' : 'text-slate-600'}`}>
            {getGCL()}
          </span>
        </div>
      </div>

      <h2 className="text-2xl font-bold text-brand-black mb-8">Live Echo Loop Demo</h2>

      <div className="flex flex-col md:flex-row items-center gap-12">
        {/* Left: Interaction Bubble */}
        <div className="relative">
          <button 
            onClick={handleStartDemo}
            disabled={phase !== DemoPhase.IDLE}
            className={`w-32 h-32 md:w-40 md:h-40 rounded-full flex items-center justify-center transition-all duration-500 relative z-10 
              ${phase === DemoPhase.IDLE ? 'bg-gradient-to-br from-brand-blue to-brand-peach hover:scale-105 shadow-[0_0_40px_rgba(186,230,253,0.5)] cursor-pointer' : ''}
              ${phase === DemoPhase.LISTENING ? 'bg-brand-blue shadow-[0_0_60px_rgba(186,230,253,0.6)] animate-pulse' : ''}
              ${phase === DemoPhase.PROCESSING ? 'bg-slate-100 animate-spin border-4 border-t-brand-blue border-r-brand-peach border-b-green-300 border-l-transparent' : ''}
              ${phase === DemoPhase.PLAYBACK ? 'bg-green-400 shadow-[0_0_60px_rgba(74,222,128,0.6)]' : ''}
            `}
          >
            {phase === DemoPhase.IDLE && <Play className="w-12 h-12 text-white ml-1" />}
            {phase === DemoPhase.LISTENING && <Mic className="w-12 h-12 text-white" />}
            {phase === DemoPhase.PROCESSING && <Activity className="w-12 h-12 text-slate-400" />}
            {phase === DemoPhase.PLAYBACK && <Activity className="w-12 h-12 text-white" />}
          </button>
          
          {/* Decorative Rings */}
          <div className={`absolute inset-0 rounded-full border border-brand-blue/30 scale-125 pointer-events-none transition-all duration-1000 ${phase !== DemoPhase.IDLE ? 'opacity-0' : 'opacity-100'}`}></div>
          <div className={`absolute inset-0 rounded-full border border-brand-peach/30 scale-150 pointer-events-none transition-all duration-1000 ${phase !== DemoPhase.IDLE ? 'opacity-0' : 'opacity-100'}`}></div>
        </div>

        {/* Right: Status Text */}
        <div className="flex-1 space-y-6">
          <div className="h-16 flex items-center">
            <p className="text-xl md:text-2xl font-light text-slate-700">
              {phase === DemoPhase.IDLE && (latencyMs ? "Loop complete. Tap again to replay." : "Tap the bubble to start the loop.")}
              {phase === DemoPhase.LISTENING && "Child speaks into Goeckoh…"}
              {phase === DemoPhase.PROCESSING && "Goeckoh is correcting and cloning in real time…"}
              {phase === DemoPhase.PLAYBACK && <span className="text-green-600 font-medium">Corrected speech plays back in the child’s own voice…</span>}
            </p>
          </div>

          <div className="space-y-3">
            <StepIndicator 
              label="Child speaks" 
              status={phase === DemoPhase.LISTENING ? 'active' : (latencyMs !== null ? 'done' : 'waiting')} 
            />
            <StepIndicator 
              label="Goeckoh corrects & clones" 
              status={phase === DemoPhase.PROCESSING ? 'active' : (phase === DemoPhase.PLAYBACK || (phase === DemoPhase.IDLE && latencyMs) ? 'done' : 'waiting')} 
            />
            <StepIndicator 
              label="Child hears corrected voice" 
              status={phase === DemoPhase.PLAYBACK ? 'active' : (phase === DemoPhase.IDLE && latencyMs ? 'done' : 'waiting')} 
            />
          </div>
          
          {audioError && (
             <div className="flex items-center gap-2 text-red-500 text-xs mt-2 bg-red-50 p-2 rounded border border-red-100">
               <AlertCircle size={14} />
               Demo audio could not be played. Check that /audio/child_input_1.m4a and /audio/echo_output_1.m4a exist.
             </div>
          )}
        </div>
      </div>
    </div>
  );
};

const StepIndicator: React.FC<{ label: string; status: 'waiting' | 'active' | 'done' }> = ({ label, status }) => {
  return (
    <div className="flex items-center gap-3">
      <div className={`w-3 h-3 rounded-full transition-colors duration-300 
        ${status === 'waiting' ? 'bg-slate-200' : ''}
        ${status === 'active' ? 'bg-brand-blue shadow-[0_0_10px_#BAE6FD]' : ''}
        ${status === 'done' ? 'bg-green-400' : ''}
      `} />
      <span className={`text-sm transition-colors duration-300 ${status === 'waiting' ? 'text-slate-400' : 'text-slate-700 font-medium'}`}>
        {label}
      </span>
    </div>
  );
};

export default EchoDemo;