export function startWatchdog(getLastFrameTime: () => number, restartWorker: () => void) {
  let lastSeen = performance.now();
  setInterval(() => {
    const t = getLastFrameTime();
    if (performance.now() - t > 200) {
      console.warn('Watchdog detected stall, restarting worker');
      restartWorker();
    }
  }, 1000);
}
