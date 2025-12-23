const regret = replayLogs ? (replayLogs[t]?.regret ?? (Math.random()-0.5) * 0.3) : (Math.random() - 0.5) * 0.5;
